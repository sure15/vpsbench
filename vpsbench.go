package main

import (
	"context"
	"crypto/sha256"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type result struct {
	Name   string
	Value  string
	Note   string
	Score  float64
	Weight float64
}

const (
	baseTotalScore = 1000.0

	cpuWeight     = 45.0
	memoryWeight  = 20.0
	diskWeight    = 25.0
	networkWeight = 10.0

	// Reference profile: a basic 1 vCPU / 1 GiB VPS scores about 1000 total.
	baseCPUFloatSingleOps = 25_000_000.0
	baseCPUFloatMultiOps  = 25_000_000.0
	baseCPUIntOps         = 350_000_000.0
	baseCPUHashMBps       = 1000.0
	baseMemoryGBps        = 4.0
	baseDiskMBps          = 200.0
	baseNetworkMbps       = 100.0

	cpuFloatSingleWeight = 35.0
	cpuFloatMultiWeight  = 35.0
	cpuIntWeight         = 15.0
	cpuHashWeight        = 15.0
)

func main() {
	var (
		duration = flag.Duration("duration", 5*time.Second, "CPU/memory/network test duration, for example 3s or 10s")
		fileMB   = flag.Int("file-mb", 512, "temporary file size for disk test, in MiB")
		dir      = flag.String("dir", os.TempDir(), "directory used for disk test")
		netURL   = flag.String("net-url", "https://speed.cloudflare.com/__down?bytes=200000000", "HTTP download URL for network test")
		skipNet  = flag.Bool("skip-net", false, "skip HTTP download test")
		skipDisk = flag.Bool("skip-disk", false, "skip disk read/write test")
	)
	flag.Parse()

	if *duration <= 0 {
		fail("duration must be greater than 0")
	}
	if *fileMB <= 0 {
		fail("file-mb must be greater than 0")
	}

	fmt.Println("VPS Benchmark")
	fmt.Println(strings.Repeat("=", 64))
	fmt.Printf("Time       : %s\n", time.Now().Format(time.RFC3339))
	fmt.Printf("OS/Arch    : %s/%s\n", runtime.GOOS, runtime.GOARCH)
	fmt.Printf("Go         : %s\n", runtime.Version())
	fmt.Printf("CPU cores  : %d\n", runtime.NumCPU())
	fmt.Printf("GOMAXPROCS : %d\n", runtime.GOMAXPROCS(0))
	fmt.Printf("Score base : %.0f points = 1 vCPU / 1 GiB reference VPS\n", baseTotalScore)
	fmt.Println(strings.Repeat("=", 64))

	results := benchCPUResults(*duration)
	results = append(results, benchMemory(*duration))

	if !*skipDisk {
		results = append(results, benchDisk(*dir, *fileMB))
	}
	if !*skipNet {
		results = append(results, benchNetwork(*netURL, *duration))
	}

	fmt.Println()
	printResults(results)
}

func benchCPUResults(duration time.Duration) []result {
	results := []result{
		benchCPUFloatSingle(duration),
		benchCPUFloatMulti(duration),
		benchCPUInt(duration),
		benchCPUHash(duration),
	}

	total := weightedScore(results)
	for i := range results {
		results[i].Weight = 0
	}
	results = append(results, result{
		Name:   "CPU TOTAL",
		Value:  grade(total),
		Note:   "combined CPU score",
		Score:  total,
		Weight: cpuWeight,
	})
	return results
}

func benchCPUFloatSingle(duration time.Duration) result {
	ops := runFloatWorker(time.Now().Add(duration))
	opsPerSec := float64(ops) / duration.Seconds()
	return result{
		Name:   "CPU float single",
		Value:  fmt.Sprintf("%.2f M ops/s", opsPerSec/1_000_000),
		Note:   "sqrt/mod loop, 1 worker",
		Score:  metricScore(opsPerSec, baseCPUFloatSingleOps),
		Weight: cpuFloatSingleWeight,
	}
}

func benchCPUFloatMulti(duration time.Duration) result {
	workers := runtime.GOMAXPROCS(0)
	stopTime := time.Now().Add(duration)
	var wg sync.WaitGroup
	results := make(chan uint64, workers)

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			results <- runFloatWorker(stopTime)
		}()
	}

	wg.Wait()
	close(results)

	var total uint64
	for n := range results {
		total += n
	}

	opsPerSec := float64(total) / duration.Seconds()
	return result{
		Name:   "CPU float multi",
		Value:  fmt.Sprintf("%.2f M ops/s", opsPerSec/1_000_000),
		Note:   fmt.Sprintf("sqrt/mod loop, %d workers", workers),
		Score:  metricScore(opsPerSec, baseCPUFloatMultiOps),
		Weight: cpuFloatMultiWeight,
	}
}

func runFloatWorker(stopTime time.Time) uint64 {
	count := uint64(0)
	x := 0.0001
	for time.Now().Before(stopTime) {
		x += math.Sqrt(x + 1.2345)
		x = math.Mod(x, 1000)
		count++
	}
	if x == -1 {
		fmt.Fprint(io.Discard, x)
	}
	return count
}

func benchCPUInt(duration time.Duration) result {
	workers := runtime.GOMAXPROCS(0)
	ctx, cancel := context.WithTimeout(context.Background(), duration)
	defer cancel()

	var ops uint64
	var checksum uint64
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(seed uint64) {
			defer wg.Done()
			x := seed + 0x9e3779b97f4a7c15
			localOps := uint64(0)
			for {
				select {
				case <-ctx.Done():
					atomic.AddUint64(&ops, localOps)
					atomic.AddUint64(&checksum, x)
					return
				default:
					for j := 0; j < 1024; j++ {
						x ^= x << 13
						x ^= x >> 7
						x ^= x << 17
						x *= 0xbf58476d1ce4e5b9
					}
					localOps += 1024
				}
			}
		}(uint64(i + 1))
	}
	wg.Wait()

	opsPerSec := float64(ops) / duration.Seconds()
	return result{
		Name:   "CPU integer",
		Value:  fmt.Sprintf("%.2f M ops/s", opsPerSec/1_000_000),
		Note:   fmt.Sprintf("bitwise/multiply mix, checksum %d", checksum),
		Score:  metricScore(opsPerSec, baseCPUIntOps),
		Weight: cpuIntWeight,
	}
}

func benchCPUHash(duration time.Duration) result {
	workers := runtime.GOMAXPROCS(0)
	ctx, cancel := context.WithTimeout(context.Background(), duration)
	defer cancel()

	var blocks uint64
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(seed byte) {
			defer wg.Done()
			buf := make([]byte, 32*1024)
			for j := range buf {
				buf[j] = byte(j) ^ seed
			}

			local := uint64(0)
			for {
				select {
				case <-ctx.Done():
					atomic.AddUint64(&blocks, local)
					return
				default:
					sum := sha256.Sum256(buf)
					copy(buf[:32], sum[:])
					local++
				}
			}
		}(byte(i))
	}
	wg.Wait()

	bytes := float64(blocks) * 32 * 1024
	mbps := bytes / duration.Seconds() / 1024 / 1024
	return result{
		Name:   "CPU SHA256",
		Value:  fmt.Sprintf("%.2f MB/s", mbps),
		Note:   fmt.Sprintf("%d workers, %.0f hashes/s", workers, float64(blocks)/duration.Seconds()),
		Score:  metricScore(mbps, baseCPUHashMBps),
		Weight: cpuHashWeight,
	}
}

func benchMemory(duration time.Duration) result {
	workers := runtime.GOMAXPROCS(0)
	ctx, cancel := context.WithTimeout(context.Background(), duration)
	defer cancel()

	var bytesCopied uint64
	var checksum uint64
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(seed int) {
			defer wg.Done()
			size := 32 * 1024 * 1024
			src := make([]byte, size)
			dst := make([]byte, size)
			r := rand.New(rand.NewSource(int64(seed) + 1))
			if _, err := r.Read(src); err != nil {
				return
			}

			localBytes := uint64(0)
			localChecksum := uint64(0)
			for {
				select {
				case <-ctx.Done():
					atomic.AddUint64(&bytesCopied, localBytes)
					atomic.AddUint64(&checksum, localChecksum)
					return
				default:
					copy(dst, src)
					localBytes += uint64(size)
					localChecksum += uint64(dst[seed%size])
				}
			}
		}(i)
	}
	wg.Wait()

	gbps := float64(bytesCopied) / duration.Seconds() / 1024 / 1024 / 1024
	return result{
		Name:   "Memory copy",
		Value:  fmt.Sprintf("%.2f GB/s", gbps),
		Note:   fmt.Sprintf("%d workers, checksum %d", workers, checksum),
		Score:  metricScore(gbps, baseMemoryGBps),
		Weight: memoryWeight,
	}
}

func benchDisk(dir string, fileMB int) result {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return failedResult("Disk sequential", diskWeight, err)
	}

	path := filepath.Join(dir, fmt.Sprintf("vpsbench-%d.tmp", time.Now().UnixNano()))
	defer os.Remove(path)

	size := int64(fileMB) * 1024 * 1024
	buf := make([]byte, 4*1024*1024)
	for i := range buf {
		buf[i] = byte(i * 31)
	}

	writeStart := time.Now()
	f, err := os.OpenFile(path, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0644)
	if err != nil {
		return failedResult("Disk sequential", diskWeight, err)
	}
	written := int64(0)
	for written < size {
		n := int64(len(buf))
		if remain := size - written; remain < n {
			n = remain
		}
		if _, err := f.Write(buf[:n]); err != nil {
			f.Close()
			return failedResult("Disk sequential", diskWeight, err)
		}
		written += n
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return failedResult("Disk sequential", diskWeight, err)
	}
	if err := f.Close(); err != nil {
		return failedResult("Disk sequential", diskWeight, err)
	}
	writeElapsed := time.Since(writeStart)

	readStart := time.Now()
	f, err = os.Open(path)
	if err != nil {
		return failedResult("Disk sequential", diskWeight, err)
	}
	read, err := io.Copy(io.Discard, f)
	closeErr := f.Close()
	if err != nil {
		return failedResult("Disk sequential", diskWeight, err)
	}
	if closeErr != nil {
		return failedResult("Disk sequential", diskWeight, closeErr)
	}
	readElapsed := time.Since(readStart)

	writeMBs := float64(written) / writeElapsed.Seconds() / 1024 / 1024
	readMBs := float64(read) / readElapsed.Seconds() / 1024 / 1024
	avgMBs := (writeMBs + readMBs) / 2
	return result{
		Name:   "Disk sequential",
		Value:  fmt.Sprintf("write %.2f MB/s, read %.2f MB/s", writeMBs, readMBs),
		Note:   fmt.Sprintf("%d MiB file in %s", fileMB, dir),
		Score:  metricScore(avgMBs, baseDiskMBps),
		Weight: diskWeight,
	}
}

func benchNetwork(rawURL string, duration time.Duration) result {
	req, err := http.NewRequest(http.MethodGet, rawURL, nil)
	if err != nil {
		return failedResult("HTTP download", networkWeight, err)
	}
	req.Header.Set("User-Agent", "vpsbench/1.0")

	client := &http.Client{Timeout: duration + 15*time.Second}
	start := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		return failedResult("HTTP download", networkWeight, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return result{Name: "HTTP download", Value: "failed", Note: resp.Status, Weight: networkWeight}
	}

	ctx, cancel := context.WithTimeout(context.Background(), duration)
	defer cancel()
	limited := &contextReader{ctx: ctx, r: resp.Body}
	n, err := io.Copy(io.Discard, limited)
	elapsed := time.Since(start)
	if err != nil && ctx.Err() == nil {
		return failedResult("HTTP download", networkWeight, err)
	}
	if elapsed <= 0 {
		elapsed = time.Nanosecond
	}

	mbps := float64(n) * 8 / elapsed.Seconds() / 1000 / 1000
	return result{
		Name:   "HTTP download",
		Value:  fmt.Sprintf("%.2f Mbps", mbps),
		Note:   fmt.Sprintf("%.2f MiB from %s", float64(n)/1024/1024, rawURL),
		Score:  metricScore(mbps, baseNetworkMbps),
		Weight: networkWeight,
	}
}

type contextReader struct {
	ctx context.Context
	r   io.Reader
}

func (r *contextReader) Read(p []byte) (int, error) {
	select {
	case <-r.ctx.Done():
		return 0, r.ctx.Err()
	default:
		return r.r.Read(p)
	}
}

func printResults(results []result) {
	fmt.Printf("%-18s %-48s %-10s %s\n", "Test", "Result", "Score", "Note")
	fmt.Println(strings.Repeat("-", 108))
	for _, r := range results {
		fmt.Printf("%-18s %-48s %-10.0f %s\n", r.Name, r.Value, r.Score, r.Note)
	}

	fmt.Println(strings.Repeat("-", 108))
	total, weight := totalScore(results)
	fmt.Printf("%-18s %-48s %-10.0f %s\n", "TOTAL", grade(total), total, fmt.Sprintf("weighted score, %.0f active weight", weight))
	fmt.Println()
	fmt.Println("Reference: 1000 points ~= basic 1 vCPU / 1 GiB VPS")
	fmt.Printf("CPU base : float single %.0f M ops/s, float multi %.0f M ops/s, integer %.0f M ops/s, SHA256 %.0f MB/s\n",
		baseCPUFloatSingleOps/1_000_000, baseCPUFloatMultiOps/1_000_000, baseCPUIntOps/1_000_000, baseCPUHashMBps)
	fmt.Printf("Other    : memory %.1f GB/s, disk %.0f MB/s avg, network %.0f Mbps\n",
		baseMemoryGBps, baseDiskMBps, baseNetworkMbps)
}

func metricScore(value, baseline float64) float64 {
	if baseline <= 0 || value <= 0 {
		return 0
	}
	return value / baseline * baseTotalScore
}

func failedResult(name string, weight float64, err error) result {
	return result{Name: name, Value: "failed", Note: err.Error(), Weight: weight}
}

func weightedScore(results []result) float64 {
	score, weight := totalScore(results)
	if weight == 0 {
		return 0
	}
	return score
}

func totalScore(results []result) (float64, float64) {
	var weightedScore float64
	var totalWeight float64
	for _, r := range results {
		if r.Weight <= 0 {
			continue
		}
		weightedScore += r.Score * r.Weight
		totalWeight += r.Weight
	}
	if totalWeight == 0 {
		return 0, 0
	}
	return weightedScore / totalWeight, totalWeight
}

func grade(score float64) string {
	switch {
	case score >= 5000:
		return "excellent"
	case score >= 2500:
		return "very good"
	case score >= 1200:
		return "good"
	case score >= 800:
		return "basic"
	default:
		return "low"
	}
}

func fail(msg string) {
	fmt.Fprintln(os.Stderr, "error:", msg)
	os.Exit(2)
}
