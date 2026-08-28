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

"""Tests for the AFD dual-batch overlap driver and batch split.

The schedule tests need no GPU: the stages only record themselves, so the
ordering invariants are checked against the resulting op log.  The split tests
do run paddle ops.
"""

import unittest

from fastdeploy.model_executor.dual_batch_overlap.dbo_runner import (
    DBOMicroState,
    assert_supports_dbo,
    run_dbo_pipeline,
)


class FakeDBOLayer:
    supports_dbo = True

    def __init__(self, layer_id, log):
        self.layer_id = layer_id
        self.log = log

    def _record(self, st, stage):
        self.log.append((st.microbatch_id, self.layer_id, stage))

    def dbo_attn(self, st):
        self._record(st, "attn")

    def dbo_dispatch_send(self, st):
        self._record(st, "dispatch_send")

    def dbo_dispatch_recv(self, st):
        self._record(st, "dispatch_recv")

    def dbo_local(self, st):
        self._record(st, "local")

    def dbo_combine_send(self, st):
        self._record(st, "combine_send")

    def dbo_combine_recv(self, st):
        self._record(st, "combine_recv")


def run(num_layers):
    log = []
    layers = [FakeDBOLayer(i, log) for i in range(num_layers)]
    run_dbo_pipeline(layers, DBOMicroState(None, None, None, 0), DBOMicroState(None, None, None, 1))
    return log


class TestDBOSchedule(unittest.TestCase):
    def test_interleaved_order(self):
        """B trails A by one op, A first in every step."""
        log = run(num_layers=2)
        expected_head = [
            (0, 0, "attn"),
            (0, 0, "dispatch_send"),
            (1, 0, "attn"),
            (0, 0, "dispatch_recv"),
            (1, 0, "dispatch_send"),
            (0, 0, "local"),
            (1, 0, "dispatch_recv"),
            (0, 0, "combine_send"),
            (1, 0, "local"),
            (0, 0, "combine_recv"),
            (1, 0, "combine_send"),
            (0, 1, "attn"),
            (1, 0, "combine_recv"),
        ]
        self.assertEqual(log[: len(expected_head)], expected_head)

    def test_every_op_runs_exactly_once(self):
        num_layers = 4
        log = run(num_layers)
        self.assertEqual(len(log), 2 * num_layers * 6)
        self.assertEqual(len(set(log)), len(log))

    def test_communication_is_covered_by_compute(self):
        """Each send is followed by at least one compute op before its recv."""
        num_layers = 3
        log = run(num_layers)
        compute = {"attn", "local"}
        drain_tail = (1, num_layers - 1, "combine_send")
        for i, (mb, layer, stage) in enumerate(log):
            if not stage.endswith("_send") or (mb, layer, stage) == drain_tail:
                continue
            recv_stage = stage.replace("_send", "_recv")
            recv_at = log.index((mb, layer, recv_stage))
            covered = [op for op in log[i + 1 : recv_at] if op[2] in compute]
            self.assertTrue(covered, f"{log[i]} has no compute between send and recv")

        # The exposed tail must be exactly that one send, i.e. the pipeline only
        # loses coverage while draining, never in steady state.
        self.assertEqual(log[-2:], [drain_tail, (1, num_layers - 1, "combine_recv")])

    def test_sends_are_consumed_before_the_next_one(self):
        """Only one transfer is in flight at a time, and its recv is the matching one."""
        for num_layers in (1, 2, 3, 8):
            in_flight = None
            for mb, layer, stage in run(num_layers):
                if stage.endswith("_send"):
                    self.assertIsNone(in_flight, f"num_layers={num_layers}: {(mb, layer, stage)} overlaps {in_flight}")
                    in_flight = (mb, layer, stage)
                elif stage.endswith("_recv"):
                    expected = (mb, layer, stage.replace("_recv", "_send"))
                    self.assertEqual(in_flight, expected, f"num_layers={num_layers}")
                    in_flight = None
            self.assertIsNone(in_flight, f"num_layers={num_layers}: unconsumed send")

    def test_missing_stage_raises(self):
        class NoStages:
            pass

        with self.assertRaises(NotImplementedError):
            assert_supports_dbo([NoStages()])

        class Partial:
            supports_dbo = True

            def dbo_attn(self, st):
                pass

        with self.assertRaises(NotImplementedError):
            assert_supports_dbo([Partial()])


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
