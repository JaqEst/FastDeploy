package manager

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/PaddlePaddle/FastDeploy/router/pkg/logger"
	"github.com/gin-gonic/gin"
)

const recoverCommitPath = "/v1/recover/commit"
const recoverTimeout = 60 * time.Second

type RecoverCommitRequest struct {
	Ranks   []int    `json:"ranks"`
	Targets []string `json:"targets"`
}

type RecoverCommitBroadcastRequest struct {
	Ranks []int `json:"ranks"`
}

type RecoverCommitResult struct {
	URL        string `json:"url"`
	StatusCode int    `json:"status_code,omitempty"`
	Body       string `json:"body,omitempty"`
	Error      string `json:"error,omitempty"`
}

func GetRecoverCommitURLs(ctx context.Context, targets []string) []string {
	if DefaultManager == nil {
		return []string{}
	}

	DefaultManager.mu.RLock()
	defer DefaultManager.mu.RUnlock()

	urlSet := make(map[string]struct{})
	addWorkerMap := func(workerMap map[string]*WorkerInfo) {
		for _, worker := range workerMap {
			if worker == nil || worker.Url == "" {
				continue
			}
			urlSet[worker.Url] = struct{}{}
		}
	}

	for _, target := range targets {
		switch strings.ToLower(strings.TrimSpace(target)) {
		case "mixed":
			addWorkerMap(DefaultManager.mixedWorkerMap)
		case "prefill":
			addWorkerMap(DefaultManager.prefillWorkerMap)
		case "decode", "attn":
			addWorkerMap(DefaultManager.decodeWorkerMap)
		case "ffn":
			addWorkerMap(DefaultManager.ffnWorkerMap)
		default:
			logger.Warn(ctx, "Ignore unknown recover commit target: %s", target)
		}
	}

	urls := make([]string, 0, len(urlSet))
	for url := range urlSet {
		urls = append(urls, url)
	}
	sort.Strings(urls)
	return urls
}

func RecoverCommit(c *gin.Context) {
	bodyBytes, err := io.ReadAll(c.Request.Body)
	if err != nil {
		logger.Error(c.Request.Context(), "Failed to read recover commit request body: %v", err)
		c.JSON(http.StatusBadRequest, gin.H{
			"code": http.StatusBadRequest,
			"msg":  "invalid request body",
		})
		return
	}
	if len(bodyBytes) == 0 {
		bodyBytes = []byte("{}")
	}

	var commitRequest RecoverCommitRequest
	if err := json.Unmarshal(bodyBytes, &commitRequest); err != nil {
		logger.Error(c.Request.Context(), "Failed to parse recover commit request body: %v", err)
		c.JSON(http.StatusBadRequest, gin.H{
			"code": http.StatusBadRequest,
			"msg":  "invalid json body",
		})
		return
	}
	if len(commitRequest.Ranks) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{
			"code": http.StatusBadRequest,
			"msg":  "ranks is required",
		})
		return
	}
	if len(commitRequest.Targets) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{
			"code": http.StatusBadRequest,
			"msg":  "targets is required",
		})
		return
	}

	targetURLs := GetRecoverCommitURLs(c.Request.Context(), commitRequest.Targets)
	if len(targetURLs) == 0 {
		c.JSON(http.StatusOK, gin.H{
			"code":         http.StatusOK,
			"msg":          "Recover commit skipped: no registered targets",
			"target_count": 0,
		})
		return
	}

	downstreamBody, err := json.Marshal(RecoverCommitBroadcastRequest{Ranks: commitRequest.Ranks})
	if err != nil {
		logger.Error(c.Request.Context(), "Failed to marshal recover commit broadcast body: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"code": http.StatusInternalServerError,
			"msg":  "failed to prepare broadcast request",
		})
		return
	}

	// Return to the caller first, then continue the broadcast in the background.
	requestBody := append([]byte(nil), downstreamBody...)
	targetCopy := append([]string(nil), targetURLs...)
	c.JSON(http.StatusAccepted, gin.H{
		"code":         http.StatusAccepted,
		"msg":          "Recover commit accepted",
		"target_count": len(targetCopy),
	})

	go launchRecoverCommitBroadcast(c.Request.Context(), requestBody, targetCopy)
}

func launchRecoverCommitBroadcast(ctx context.Context, broadcastBody []byte, targetURLs []string) {
	// Keep request values from the incoming context, but do not let cancellation stop the background work.
	bgCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), recoverTimeout)
	defer cancel()

	results := broadcastRecoverCommit(bgCtx, broadcastBody, targetURLs)
	hasFailure := false
	for _, result := range results {
		if result.Error != "" || result.StatusCode < http.StatusOK || result.StatusCode >= http.StatusMultipleChoices {
			hasFailure = true
			break
		}
	}

	if hasFailure {
		logger.Warn(bgCtx, "Recover commit broadcast partially failed: %v", results)
		return
	}
	logger.Info(bgCtx, "Recover commit broadcast complete: %d targets", len(results))
}

func broadcastRecoverCommit(
	ctx context.Context,
	bodyBytes []byte,
	targetURLs []string,
) []RecoverCommitResult {
	client := &http.Client{Timeout: recoverTimeout}
	resultsCh := make(chan RecoverCommitResult, len(targetURLs))
	var wg sync.WaitGroup

	for _, targetURL := range targetURLs {
		wg.Add(1)
		go func(baseURL string) {
			defer wg.Done()
			resultsCh <- postRecoverCommit(ctx, client, baseURL, bodyBytes)
		}(targetURL)
	}

	go func() {
		wg.Wait()
		close(resultsCh)
	}()

	results := make([]RecoverCommitResult, 0, len(targetURLs))
	for result := range resultsCh {
		results = append(results, result)
	}
	sort.Slice(results, func(i, j int) bool { return results[i].URL < results[j].URL })
	return results
}

func postRecoverCommit(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	bodyBytes []byte,
) RecoverCommitResult {
	url := strings.TrimRight(baseURL, "/") + recoverCommitPath
	result := RecoverCommitResult{URL: url}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(bodyBytes))
	if err != nil {
		result.Error = err.Error()
		return result
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		result.Error = err.Error()
		return result
	}
	defer resp.Body.Close()

	result.StatusCode = resp.StatusCode
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		result.Error = err.Error()
		return result
	}
	result.Body = string(respBody)
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		result.Error = resp.Status
	}
	return result
}
