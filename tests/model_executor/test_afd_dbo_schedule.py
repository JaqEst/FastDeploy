# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
import unittest
from types import SimpleNamespace

import paddle

from fastdeploy.model_executor.dual_batch_overlap.dbo_wrapper import (
    DBOWrapper,
    dbo_enabled,
    dbo_maybe_run_recv_hook,
    dbo_register_recv_hook,
    dbo_yield,
)


def make_config(enable_dbo=True, is_ffn=False):
    """The three fd_config fields DBOWrapper reads."""
    return SimpleNamespace(
        afd_config=SimpleNamespace(enable_dbo=enable_dbo, is_ffn=is_ffn),
        speculative_config=SimpleNamespace(enabled_speculative_decoding=lambda: False),
    )


class FakeModel:
    """A fake model that records the order of its calls."""

    STAGES = (
        "attn_route",
        "dispatch_send",
        "dispatch_recv",
        "local",
        "combine_send",
        "combine_recv",
    )

    def __init__(self, log, num_layers, fail_on=None):
        self.log = log
        self.num_layers = num_layers
        self.fail_on = fail_on

    def _record(self, micro_batch_id, layer_id, stage):
        self.log.append((micro_batch_id, layer_id, stage))
        if self.fail_on == (micro_batch_id, layer_id, stage):
            raise RuntimeError("stage failed")

    def _comm(self, micro_batch_id, layer_id, phase):
        dbo_maybe_run_recv_hook()
        self._record(micro_batch_id, layer_id, f"{phase}_send")
        dbo_register_recv_hook(lambda: self._record(micro_batch_id, layer_id, f"{phase}_recv"))
        if not dbo_enabled():
            # No peer to hand the wait to: the transfer completes inline.
            self._record(micro_batch_id, layer_id, f"{phase}_recv")
        dbo_yield()

    def __call__(self, ids_remove_padding=None, forward_meta=None):
        micro_batch_id = {"thread0": 0, "thread1": 1}.get(threading.current_thread().name, 0)
        for layer_id in range(self.num_layers):
            self._record(micro_batch_id, layer_id, "attn_route")
            self._comm(micro_batch_id, layer_id, "dispatch")
            self._record(micro_batch_id, layer_id, "local")
            self._comm(micro_batch_id, layer_id, "combine")
        return paddle.zeros([1], dtype="float32")


def make_forward_meta():
    from fastdeploy.model_executor.forward_meta import ForwardMeta

    return ForwardMeta(
        ids_remove_padding=paddle.zeros([2], dtype="int64"),
        dbo_micro_inputs=[{}, {}],
    )


def run(num_layers, fail_on=None):
    log = []
    wrapper = DBOWrapper(FakeModel(log, num_layers, fail_on=fail_on), make_config())
    wrapper(ids_remove_padding=None, forward_meta=make_forward_meta())
    return log


class TestDBOSchedule(unittest.TestCase):
    def test_b_trails_a_by_one_section(self):
        """A is one section ahead, then the two threads strictly take turns."""
        log = run(num_layers=2)
        expected_head = [
            (0, 0, "attn_route"),
            (0, 0, "dispatch_send"),
            (1, 0, "attn_route"),
            (0, 0, "dispatch_recv"),
            (1, 0, "dispatch_send"),
            (0, 0, "local"),
            (1, 0, "dispatch_recv"),
            (0, 0, "combine_send"),
            (1, 0, "local"),
            (0, 0, "combine_recv"),
            (1, 0, "combine_send"),
            (0, 1, "attn_route"),
            (1, 0, "combine_recv"),
        ]
        self.assertEqual(log[: len(expected_head)], expected_head)

    def test_only_one_transfer_is_in_flight(self):
        """Both micro-batches share one DeepEP low-latency buffer.

        A send must be consumed by its own recv before either thread starts the
        next one, otherwise the second transfer overwrites the first's receive
        buffer.
        """
        for num_layers in (1, 2, 3, 8):
            in_flight = None
            for mb, layer, stage in run(num_layers):
                if stage.endswith("_send"):
                    self.assertIsNone(
                        in_flight,
                        f"num_layers={num_layers}: {(mb, layer, stage)} starts while {in_flight} is unconsumed",
                    )
                    in_flight = (mb, layer, stage)
                elif stage.endswith("_recv"):
                    expected = (mb, layer, stage.replace("_recv", "_send"))
                    self.assertEqual(in_flight, expected, f"num_layers={num_layers}")
                    in_flight = None
            self.assertIsNone(in_flight, f"num_layers={num_layers}: unconsumed send")

    def test_transfers_are_covered_by_peer_compute(self):
        """Each send has real peer compute, not just a launch, before its recv."""
        compute = {"attn_route", "local"}
        num_layers = 3
        log = run(num_layers)
        # B's final combine is the drain tail: A has already finished, so there is
        # no peer left to cover it.
        drain_tail = (1, num_layers - 1, "combine_send")
        for i, (mb, layer, stage) in enumerate(log):
            if not stage.endswith("_send") or (mb, layer, stage) == drain_tail:
                continue
            recv_at = log.index((mb, layer, stage.replace("_send", "_recv")))
            covered = [op for op in log[i + 1 : recv_at] if op[2] in compute]
            self.assertTrue(covered, f"{log[i]} has no peer compute before its recv")

        # The exposed tail is exactly that one transfer, i.e. coverage is only lost
        # while draining, never in steady state.
        self.assertEqual(log[-2:], [drain_tail, (1, num_layers - 1, "combine_recv")])

    def test_every_section_runs_exactly_once(self):
        num_layers = 4
        log = run(num_layers)
        self.assertEqual(len(log), 2 * num_layers * len(FakeModel.STAGES))
        self.assertEqual(len(set(log)), len(log))

    def test_failure_propagates_without_hanging(self):
        """A dead thread must release its peer, not park it forever."""
        with self.assertRaises(RuntimeError):
            run(num_layers=2, fail_on=(0, 0, "local"))

        with self.assertRaises(RuntimeError):
            run(num_layers=2, fail_on=(1, 1, "attn_route"))

    def test_dbo_runs_two_micro_batches_without_splitting_on_the_ffn_role(self):
        """The FFN role owns no tokens but must still issue two micro-batches."""
        log = []
        wrapper = DBOWrapper(FakeModel(log, num_layers=2), make_config(is_ffn=True))
        forward_meta = make_forward_meta()
        forward_meta.dbo_micro_inputs = None
        wrapper(ids_remove_padding=None, forward_meta=forward_meta)
        self.assertEqual(sorted(mb for mb, _, _ in log), sorted([0, 1] * 2 * len(FakeModel.STAGES)))

    def test_missing_micro_inputs_is_an_error_on_a_token_owning_role(self):
        log = []
        wrapper = DBOWrapper(FakeModel(log, num_layers=1), make_config())
        forward_meta = make_forward_meta()
        forward_meta.dbo_micro_inputs = None
        with self.assertRaises(AssertionError):
            wrapper(ids_remove_padding=None, forward_meta=forward_meta)

    def test_dbo_is_skipped_when_disabled(self):
        """Without DBO the transfers complete inline, in one micro-batch."""
        log = []
        wrapper = DBOWrapper(FakeModel(log, num_layers=1), make_config(enable_dbo=False))
        wrapper(ids_remove_padding=None, forward_meta=make_forward_meta())
        self.assertEqual([mb for mb, _, _ in log], [0] * len(FakeModel.STAGES))


class TestDBOSplit(unittest.TestCase):
    """Slot metadata and token payload must stay aligned, including empty slots."""

    def _split(self, seq_lens_this_time, token_buffers=None, split_token_num=None):
        import paddle

        from fastdeploy.model_executor.dual_batch_overlap.dbo_split import (
            allocate_dbo_token_buffer,
            build_dbo_micro_inputs,
            split_decode_forward_meta,
        )
        from fastdeploy.model_executor.forward_meta import ForwardMeta
        from fastdeploy.model_executor.pre_and_post_process import pre_process

        num_seqs = len(seq_lens_this_time)
        max_len = 8
        # Token value encodes its owning slot, so misassignment is visible.
        input_ids = (paddle.arange(num_seqs, dtype="int64") * max_len).unsqueeze(-1).tile([1, max_len])
        seq_lens = paddle.to_tensor(seq_lens_this_time, dtype="int32")
        seq_lens_decoder = paddle.to_tensor([(i + 1) * 10 for i in range(num_seqs)], dtype="int32")
        # The runner pads the token count up to a captured graph shape before
        # pre_process, so the full-batch tensors already carry that padding.
        token_num = sum(seq_lens_this_time) if split_token_num is None else split_token_num
        ids, bid, cu_q, cu_k, _, _, _ = pre_process(token_num, input_ids, seq_lens, False)

        buffers = [{"decoder_batch_ids": f"buf{i}"} for i in range(2)]
        if token_buffers is None:
            token_buffers = [allocate_dbo_token_buffer(num_seqs, max_len) for _ in range(2)]
        meta = ForwardMeta(
            ids_remove_padding=ids,
            seq_lens_this_time=seq_lens,
            batch_id_per_token=bid,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            dbo_micro_inputs=build_dbo_micro_inputs(
                ids,
                bid,
                seq_lens,
                seq_lens_decoder,
                cu_q,
                token_num,
                buffers,
                token_buffers,
            ),
        )
        return split_decode_forward_meta(meta)

    def test_token_payload_matches_slot_metadata(self):
        for seq_lens in ([1, 1, 1, 1], [1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 1, 1], [1], [0, 0]):
            meta_a, meta_b = self._split(seq_lens)
            msg = f"seq_lens={seq_lens}"
            for meta in (meta_a, meta_b):
                slt = meta.seq_lens_this_time.numpy()
                # Full-length slot arrays keep batch_id_per_token in the global space.
                self.assertEqual(len(slt), len(seq_lens), msg)
                self.assertEqual(int(slt.sum()), meta.ids_remove_padding.shape[0], msg)
                # A slot this micro-batch does not own must look idle to attention.
                dec = meta.seq_lens_decoder.numpy()
                self.assertTrue(all(d == 0 for length, d in zip(slt, dec) if length == 0), msg)
                # Every token must sit at the offset its own slot declares.
                cu = meta.cu_seqlens_q.numpy()
                for tok_idx, slot in enumerate(meta.batch_id_per_token.numpy().tolist()):
                    self.assertEqual(cu[slot], tok_idx, msg)
                    self.assertEqual(meta.ids_remove_padding.numpy()[tok_idx], slot * 8, msg)
            # Together the two micro-batches cover the batch exactly once.
            self.assertEqual(
                (meta_a.seq_lens_this_time + meta_b.seq_lens_this_time).numpy().tolist(), seq_lens, msg
            )

    def test_token_counts_are_balanced(self):
        for seq_lens, expected in (([1, 1, 1, 1], (2, 2)), ([1, 0, 1, 1], (2, 1)), ([1], (1, 0))):
            meta_a, meta_b = self._split(seq_lens)
            got = (meta_a.ids_remove_padding.shape[0], meta_b.ids_remove_padding.shape[0])
            self.assertEqual(got, expected, f"seq_lens={seq_lens}")

    def test_attn_buffers_are_not_shared(self):
        meta_a, meta_b = self._split([1, 1, 1, 1])
        self.assertEqual(meta_a.decoder_batch_ids, "buf0")
        self.assertEqual(meta_b.decoder_batch_ids, "buf1")

    def test_token_buffer_addresses_are_stable(self):
        # CUDA Graph replay reads the addresses recorded at capture, so a changing
        # token count must not move the buffers.
        from fastdeploy.model_executor.dual_batch_overlap.dbo_split import (
            allocate_dbo_token_buffer,
        )

        token_buffers = [allocate_dbo_token_buffer(4, 8) for _ in range(2)]
        addresses = None
        for seq_lens in ([1, 1, 1, 1], [1, 0, 1, 0], [1, 1, 1, 1], [0, 0, 0, 1]):
            metas = self._split(seq_lens, token_buffers=token_buffers)
            got = [
                getattr(meta, name).data_ptr()
                for meta in metas
                for name in ("seq_lens_this_time", "ids_remove_padding", "batch_id_per_token", "cu_seqlens_q")
            ]
            if addresses is not None:
                self.assertEqual(got, addresses, f"seq_lens={seq_lens}")
            addresses = got

    def test_padded_split_keeps_shapes_constant(self):
        # A replayed graph carries the split it was captured with. Splitting on the
        # padded shape has to give the same micro-batch shapes for every real token
        # count that maps to it, otherwise the graph reads past what was written.
        shapes = None
        for seq_lens in ([1, 1, 1, 1], [1, 1, 1, 0], [1, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]):
            meta_a, meta_b = self._split(seq_lens, split_token_num=4)
            msg = f"seq_lens={seq_lens}"
            got = (meta_a.ids_remove_padding.shape[0], meta_b.ids_remove_padding.shape[0])
            self.assertEqual(got, (2, 2), msg)
            if shapes is not None:
                self.assertEqual(got, shapes, msg)
            shapes = got
            # Padding rows are never referenced: cu_seqlens only spans real tokens.
            for meta in (meta_a, meta_b):
                self.assertEqual(int(meta.seq_lens_this_time.numpy().sum()), int(meta.cu_seqlens_q.numpy()[-1]), msg)
            self.assertEqual(
                (meta_a.seq_lens_this_time + meta_b.seq_lens_this_time).numpy().tolist(), seq_lens, msg
            )


if __name__ == "__main__":
    unittest.main()
