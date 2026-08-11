package manager

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/PaddlePaddle/FastDeploy/router/internal/config"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
)

type recoverCommitHTTPResponse struct {
	Code        int                   `json:"code"`
	Msg         string                `json:"msg"`
	TargetCount int                   `json:"target_count"`
	Results     []RecoverCommitResult `json:"results"`
}

func decodeRecoverCommitHTTPResponse(t *testing.T, body string) recoverCommitHTTPResponse {
	t.Helper()

	var resp recoverCommitHTTPResponse
	if err := json.Unmarshal([]byte(body), &resp); err != nil {
		t.Fatalf("failed to decode recover commit response: %v", err)
	}
	return resp
}

func TestGetRecoverCommitURLsDeduplicatesSortsAndFiltersByNodeType(t *testing.T) {
	Init(&config.Config{})
	DefaultManager.prefillWorkerMap = map[string]*WorkerInfo{
		"prefill": {Url: "http://worker-b"},
	}
	DefaultManager.decodeWorkerMap = map[string]*WorkerInfo{
		"decode": {Url: "http://worker-a"},
	}
	DefaultManager.mixedWorkerMap = map[string]*WorkerInfo{
		"mixed": {Url: "http://worker-b"},
	}
	DefaultManager.ffnWorkerMap = map[string]*WorkerInfo{
		"ffn": {Url: "http://worker-c"},
	}

	cases := []struct {
		name    string
		targets []string
		want    []string
	}{
		{
			name:    "decode and mixed dedupe to sorted urls",
			targets: []string{"decode", "mixed", "unknown"},
			want:    []string{"http://worker-a", "http://worker-b"},
		},
		{
			name:    "attn maps to decode workers and ffn is included",
			targets: []string{"attn", "ffn"},
			want:    []string{"http://worker-a", "http://worker-c"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			urls := GetRecoverCommitURLs(context.Background(), tc.targets)
			assert.Equal(t, tc.want, urls)
		})
	}
}

func TestRecoverCommitReturnsBeforeBroadcastCompletes(t *testing.T) {
	release := make(chan struct{})
	started := make(chan struct{}, 1)
	var prefillCount int32
	prefillServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/v1/recover/commit", r.URL.Path)
		body, err := io.ReadAll(r.Body)
		assert.NoError(t, err)
		assert.Equal(t, `{"ranks":[2]}`, string(body))
		atomic.AddInt32(&prefillCount, 1)
		select {
		case started <- struct{}{}:
		default:
		}
		<-release
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer prefillServer.Close()

	Init(&config.Config{Manager: config.ManagerConfig{HealthCheckTimeoutSecs: 1}})
	DefaultManager.prefillWorkerMap = map[string]*WorkerInfo{
		"prefill": {Url: prefillServer.URL},
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	body := `{"ranks":[2],"targets":["prefill"]}`
	c.Request = httptest.NewRequest("POST", "/v1/recover/commit", bytes.NewBufferString(body))

	done := make(chan struct{})
	go func() {
		RecoverCommit(c)
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("RecoverCommit did not return in time")
	}

	resp := decodeRecoverCommitHTTPResponse(t, w.Body.String())
	assert.Equal(t, http.StatusAccepted, w.Code)
	assert.Equal(t, http.StatusAccepted, resp.Code)
	assert.Equal(t, "Recover commit accepted", resp.Msg)
	assert.Equal(t, 1, resp.TargetCount)
	assert.Len(t, resp.Results, 0)

	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("background broadcast did not start")
	}

	close(release)
	assert.Eventually(t, func() bool {
		return atomic.LoadInt32(&prefillCount) == 1
	}, time.Second, 10*time.Millisecond)
}

func TestBroadcastRecoverCommitReportsPartialFailure(t *testing.T) {
	okServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer okServer.Close()
	failedServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer failedServer.Close()

	Init(&config.Config{Manager: config.ManagerConfig{HealthCheckTimeoutSecs: 1}})
	DefaultManager.prefillWorkerMap = map[string]*WorkerInfo{
		"prefill": {Url: okServer.URL},
	}
	DefaultManager.decodeWorkerMap = map[string]*WorkerInfo{
		"decode": {Url: failedServer.URL},
	}

	results := broadcastRecoverCommit(
		context.Background(),
		[]byte(`{"ranks":[2]}`),
		[]string{okServer.URL, failedServer.URL},
	)

	assert.Len(t, results, 2)

	resultsByURL := make(map[string]RecoverCommitResult, len(results))
	for _, result := range results {
		resultsByURL[result.URL] = result
	}

	okResult := resultsByURL[okServer.URL+recoverCommitPath]
	assert.Equal(t, http.StatusOK, okResult.StatusCode)
	assert.Empty(t, okResult.Body)
	assert.Empty(t, okResult.Error)

	failedResult := resultsByURL[failedServer.URL+recoverCommitPath]
	assert.Equal(t, http.StatusInternalServerError, failedResult.StatusCode)
	assert.Empty(t, failedResult.Body)
	assert.Equal(t, "500 Internal Server Error", failedResult.Error)
}

func TestRecoverCommitRequiresCoreFields(t *testing.T) {
	Init(&config.Config{})

	cases := []struct {
		name    string
		body    string
		wantMsg string
	}{
		{
			name:    "missing ranks",
			body:    `{"targets":["prefill"]}`,
			wantMsg: "ranks is required",
		},
		{
			name:    "missing targets",
			body:    `{"ranks":[2]}`,
			wantMsg: "targets is required",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			c.Request = httptest.NewRequest("POST", "/v1/recover/commit", bytes.NewBufferString(tc.body))

			RecoverCommit(c)

			resp := decodeRecoverCommitHTTPResponse(t, w.Body.String())
			assert.Equal(t, http.StatusBadRequest, w.Code)
			assert.Equal(t, http.StatusBadRequest, resp.Code)
			assert.Equal(t, tc.wantMsg, resp.Msg)
		})
	}
}
