package main

import (
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha1"
	"flag"
	"fmt"
	"hash/crc32"
	"io"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type result struct {
	Group  string
	Name   string
	Value  string
	Note   string
	Score  float64
	Weight float64
}

type progressTracker struct {
	total int
	done  int
	start time.Time
}

const baseScore = 1000.0

const (
	cpuOverallWeight    = 50.0
	memoryOverallWeight = 25.0
	diskOverallWeight   = 25.0

	cpuSingleOverallWeight = 40.0
	cpuMultiOverallWeight  = 60.0

	cpuCompressionWeight = 18.0
	cpuTextWeight        = 14.0
	cpuImageWeight       = 20.0
	cpuMLWeight          = 18.0
	cpuRenderWeight      = 18.0
	cpuCryptoWeight      = 12.0

	memoryCopyWeight   = 45.0
	memoryRandomWeight = 35.0
	memoryAllocWeight  = 20.0

	diskSeqWriteWeight = 30.0
	diskSeqReadWeight  = 30.0
	diskRandomRWWeight = 25.0
	diskMetadataWeight = 15.0
)

// Custom baselines. 1000 points roughly represents a small 1 vCPU / 1 GiB VPS.
const (
	baseCPUCompressionMBps = 100.0
	baseCPUTextMBps        = 120.0
	baseCPUImageMPix       = 200.0
	baseCPUMLGFLOPS        = 2.0
	baseCPURenderMRays     = 800.0
	baseCPUCryptoMBps      = 1000.0

	baseMemoryCopyGBps      = 4.0
	baseMemoryRandomMOps    = 18.0
	baseMemoryAllocKAllocs  = 500.0
	baseDiskSeqWriteMBps    = 120.0
	baseDiskSeqReadMBps     = 180.0
	baseDiskRandomRWMBps    = 18.0
	baseDiskMetadataKOpsSec = 6.0
)

func main() {
	var (
		duration = flag.Duration("duration", 5*time.Second, "duration for CPU and memory subtests, for example 3s or 10s")
		fileMB   = flag.Int("file-mb", 512, "temporary file size for disk sequential tests, in MiB")
		dir      = flag.String("dir", os.TempDir(), "directory used for disk tests")
		memGB    = flag.Float64("mem-gb", 0, "override detected physical memory in GiB")
		skipDisk = flag.Bool("skip-disk", false, "skip disk tests")
	)
	flag.Parse()

	if *duration <= 0 {
		fail("duration must be greater than 0")
	}
	if *fileMB <= 0 {
		fail("file-mb must be greater than 0")
	}

	memoryGB, memoryNote := detectMemoryGB(*memGB)
	totalSteps := 12 + 3
	if !*skipDisk {
		totalSteps += 4
	}
	progress := &progressTracker{total: totalSteps, start: time.Now()}

	header := benchmarkHeader(memoryGB, memoryNote)
	fmt.Print(header)

	var results []result
	results = append(results, benchCPU(*duration, progress)...)
	results = append(results, benchMemory(*duration, memoryGB, memoryNote, progress)...)
	if !*skipDisk {
		results = append(results, benchDisk(*dir, *fileMB, progress)...)
	}

	fmt.Println()
	report := renderReport(header, results)
	fmt.Print(report)
	filename := resultFilename(time.Now())
	if err := os.WriteFile(filename, []byte(report), 0644); err != nil {
		fmt.Fprintf(os.Stderr, "write result failed: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("\nResult file: %s\n", filename)
}

func benchmarkHeader(memoryGB float64, memoryNote string) string {
	var b strings.Builder
	b.WriteString("Go VPS Benchmark\n")
	b.WriteString(strings.Repeat("=", 76) + "\n")
	b.WriteString(fmt.Sprintf("Time       : %s\n", time.Now().Format(time.RFC3339)))
	b.WriteString(fmt.Sprintf("OS/Arch    : %s/%s\n", runtime.GOOS, runtime.GOARCH))
	b.WriteString(fmt.Sprintf("Go         : %s\n", runtime.Version()))
	b.WriteString(fmt.Sprintf("CPU cores  : %d\n", runtime.NumCPU()))
	b.WriteString(fmt.Sprintf("GOMAXPROCS : %d\n", runtime.GOMAXPROCS(0)))
	b.WriteString(fmt.Sprintf("Memory     : %.2f GiB (%s)\n", memoryGB, memoryNote))
	b.WriteString(fmt.Sprintf("Score base : %.0f points ~= small 1 vCPU / 1 GiB VPS reference\n", baseScore))
	b.WriteString(strings.Repeat("=", 76) + "\n")
	return b.String()
}

func (p *progressTracker) complete(name string) {
	p.done++
	if p.done > p.total {
		p.done = p.total
	}
	percent := float64(p.done) / float64(p.total) * 100
	fmt.Printf("[progress] %6.2f%% (%d/%d) completed: %s, elapsed %s\n",
		percent, p.done, p.total, name, time.Since(p.start).Round(time.Second))
}

func benchCPU(duration time.Duration, progress *progressTracker) []result {
	single := benchCPUSet(duration, 1, "CPU-1C", "single-core", progress)
	singleTotal := weightedScore(single)
	for i := range single {
		single[i].Weight = 0
	}
	single = append(single, result{
		Group: "CPU-1C", Name: "CPU SINGLE TOTAL", Value: grade(singleTotal),
		Note: "weighted single-core CPU score", Score: singleTotal,
	})

	multiWorkers := runtime.GOMAXPROCS(0)
	multi := benchCPUSet(duration, multiWorkers, "CPU-MC", "multi-core", progress)
	multiTotal := weightedScore(multi)
	for i := range multi {
		multi[i].Weight = 0
	}
	multi = append(multi, result{
		Group: "CPU-MC", Name: "CPU MULTI TOTAL", Value: grade(multiTotal),
		Note: fmt.Sprintf("weighted multi-core CPU score, %d workers", multiWorkers), Score: multiTotal,
	})

	cpuOverall := totalScoreOnly([]result{
		{Score: singleTotal, Weight: cpuSingleOverallWeight},
		{Score: multiTotal, Weight: cpuMultiOverallWeight},
	})

	results := append(single, multi...)
	results = append(results, result{
		Group: "CPU", Name: "CPU OVERALL TOTAL", Value: grade(cpuOverall),
		Note: "single-core 40% + multi-core 60%", Score: cpuOverall, Weight: cpuOverallWeight,
	})
	return results
}

func benchCPUSet(duration time.Duration, workers int, group, label string, progress *progressTracker) []result {
	results := []result{
		benchCPUCompression(duration, workers, group, label),
		benchCPUText(duration, workers, group, label),
		benchCPUImage(duration, workers, group, label),
		benchCPUML(duration, workers, group, label),
		benchCPURender(duration, workers, group, label),
		benchCPUCrypto(duration, workers, group, label),
	}
	for _, r := range results {
		progress.complete(r.Group + " " + r.Name)
	}
	return results
}

func benchCPUCompression(duration time.Duration, workers int, group, label string) result {
	data := makeBinaryCorpus(2 * 1024 * 1024)
	mbps, checksum := runParallelBytes(duration, workers, func() (int, int) {
		var out bytes.Buffer
		zw, _ := gzip.NewWriterLevel(&out, gzip.BestSpeed)
		_, _ = zw.Write(data)
		_ = zw.Close()
		sum := sha1.Sum(out.Bytes())
		return len(data), int(sum[0])
	})
	return metricResult(group, "File compression", fmt.Sprintf("%.2f MB/s", mbps),
		fmt.Sprintf("%s gzip best-speed, checksum %d", label, checksum), mbps, baseCPUCompressionMBps, cpuCompressionWeight)
}

func benchCPUText(duration time.Duration, workers int, group, label string) result {
	data := []byte(makeMarkdownCorpus(3 * 1024 * 1024))
	re := regexp.MustCompile(`(?m)(^#{1,3}\s+\w+)|(\[[^\]]+\]\([^)]+\))|(` + "`[^`]+`" + `)`)
	mbps, checksum := runParallelBytes(duration, workers, func() (int, int) {
		matches := re.FindAllIndex(data, -1)
		return len(data), len(matches)
	})
	return metricResult(group, "Text processing", fmt.Sprintf("%.2f MB/s", mbps),
		fmt.Sprintf("%s regex markdown scan, checksum %d", label, checksum), mbps, baseCPUTextMBps, cpuTextWeight)
}

func benchCPUImage(duration time.Duration, workers int, group, label string) result {
	w, h := 1024, 768
	src := make([]uint8, w*h*4)
	for i := range src {
		src[i] = uint8(i*31 + i/7)
	}

	mpix, checksum := runParallelUnits(duration, workers, int64(w*h), func() int {
		dst := make([]uint8, len(src))
		local := 0
		for y := 1; y < h-1; y++ {
			for x := 1; x < w-1; x++ {
				i := (y*w + x) * 4
				for c := 0; c < 3; c++ {
					v := int(src[i+c])*5 - int(src[i-4+c]) - int(src[i+4+c]) - int(src[i-w*4+c]) - int(src[i+w*4+c])
					if v < 0 {
						v = 0
					}
					if v > 255 {
						v = 255
					}
					dst[i+c] = uint8(v)
					local += v
				}
				dst[i+3] = 255
			}
		}
		return local + int(dst[len(dst)/2])
	})
	return metricResult(group, "Image filter", fmt.Sprintf("%.2f MPix/s", mpix/1_000_000),
		fmt.Sprintf("%s sharpen filter, checksum %d", label, checksum), mpix/1_000_000, baseCPUImageMPix, cpuImageWeight)
}

func benchCPUML(duration time.Duration, workers int, group, label string) result {
	n := 96
	a, b := make([]float64, n*n), make([]float64, n*n)
	for i := range a {
		a[i] = math.Sin(float64(i))
		b[i] = math.Cos(float64(i))
	}
	opsPerRun := int64(2 * n * n * n)
	flops, checksum := runParallelUnits(duration, workers, opsPerRun, func() int {
		c := make([]float64, n*n)
		sum := 0.0
		for i := 0; i < n; i++ {
			for k := 0; k < n; k++ {
				av := a[i*n+k]
				for j := 0; j < n; j++ {
					c[i*n+j] += av * b[k*n+j]
				}
			}
		}
		for i := 0; i < n; i += 17 {
			sum += c[i]
		}
		return int(math.Abs(sum)) & 0xffff
	})
	gflops := flops / 1_000_000_000
	return metricResult(group, "Machine learning", fmt.Sprintf("%.2f GFLOPS", gflops),
		fmt.Sprintf("%s matrix multiply kernel, checksum %d", label, checksum), gflops, baseCPUMLGFLOPS, cpuMLWeight)
}

func benchCPURender(duration time.Duration, workers int, group, label string) result {
	width, height := 320, 180
	raysPerRun := int64(width * height)
	rays, checksum := runParallelUnits(duration, workers, raysPerRun, func() int {
		local := 0
		for y := 0; y < height; y++ {
			fy := (float64(y)/float64(height))*2 - 1
			for x := 0; x < width; x++ {
				fx := (float64(x)/float64(width))*2 - 1
				dx, dy, dz := fx, fy, 1.4
				inv := 1 / math.Sqrt(dx*dx+dy*dy+dz*dz)
				dx, dy, dz = dx*inv, dy*inv, dz*inv
				b := 2 * (-3 * dz)
				c := 9 - 1.2
				disc := b*b - 4*c
				if disc > 0 {
					t := (-b - math.Sqrt(disc)) / 2
					local += int(t * 1000)
				} else {
					local += int((dx + dy + dz) * 100)
				}
			}
		}
		return local
	})
	mrays := rays / 1_000_000
	return metricResult(group, "Ray tracer", fmt.Sprintf("%.2f MRays/s", mrays),
		fmt.Sprintf("%s sphere ray intersections, checksum %d", label, checksum), mrays, baseCPURenderMRays, cpuRenderWeight)
}

func benchCPUCrypto(duration time.Duration, workers int, group, label string) result {
	data := makeBinaryCorpus(4 * 1024 * 1024)
	mbps, checksum := runParallelBytes(duration, workers, func() (int, int) {
		sum := sha1.Sum(data)
		return len(data), int(sum[0])
	})
	return metricResult(group, "Cryptography", fmt.Sprintf("%.2f MB/s", mbps),
		fmt.Sprintf("%s SHA1 throughput, checksum %d", label, checksum), mbps, baseCPUCryptoMBps, cpuCryptoWeight)
}

func benchMemory(duration time.Duration, memoryGB float64, memoryNote string, progress *progressTracker) []result {
	items := []result{
		benchMemoryCopy(duration),
		benchMemoryRandom(duration),
		benchMemoryAlloc(duration),
	}
	for _, r := range items {
		progress.complete(r.Group + " " + r.Name)
	}
	perGBTotal := weightedScore(items)
	for i := range items {
		items[i].Weight = 0
	}

	capacityScore := memoryGB * baseScore
	capacityFactor := math.Sqrt(math.Max(memoryGB, 0.25))
	total := perGBTotal * capacityFactor

	items = append(items,
		result{Group: "MEM", Name: "Memory capacity", Value: fmt.Sprintf("%.2f GiB", memoryGB), Note: memoryNote, Score: capacityScore},
		result{Group: "MEM", Name: "MEMORY PER-GB TOTAL", Value: grade(perGBTotal), Note: "bandwidth/latency/alloc score before capacity adjustment", Score: perGBTotal},
		result{Group: "MEM", Name: "MEMORY TOTAL", Value: grade(total), Note: fmt.Sprintf("per-GB score * sqrt(%.2f GiB)", memoryGB), Score: total, Weight: memoryOverallWeight},
	)
	return items
}

func benchMemoryCopy(duration time.Duration) result {
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
			for j := range src {
				src[j] = byte(j*17 + seed)
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
					localChecksum += uint64(dst[(seed*4099)%size])
				}
			}
		}(i)
	}
	wg.Wait()

	gbps := float64(bytesCopied) / duration.Seconds() / 1024 / 1024 / 1024
	return metricResult("MEM", "Memory copy", fmt.Sprintf("%.2f GB/s", gbps),
		fmt.Sprintf("%d workers, checksum %d", workers, checksum), gbps, baseMemoryCopyGBps, memoryCopyWeight)
}

func benchMemoryRandom(duration time.Duration) result {
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
			size := 8 * 1024 * 1024
			data := make([]uint64, size)
			for j := range data {
				data[j] = uint64(j) ^ seed
			}
			mask := uint64(size - 1)
			x := seed*6364136223846793005 + 1
			localOps := uint64(0)
			localChecksum := uint64(0)
			for {
				select {
				case <-ctx.Done():
					atomic.AddUint64(&ops, localOps)
					atomic.AddUint64(&checksum, localChecksum)
					return
				default:
					for j := 0; j < 4096; j++ {
						x = x*2862933555777941757 + 3037000493
						idx := x & mask
						data[idx] ^= x
						localChecksum += data[idx]
					}
					localOps += 4096
				}
			}
		}(uint64(i + 1))
	}
	wg.Wait()

	mops := float64(ops) / duration.Seconds() / 1_000_000
	return metricResult("MEM", "Memory random", fmt.Sprintf("%.2f M ops/s", mops),
		fmt.Sprintf("random read/write, checksum %d", checksum), mops, baseMemoryRandomMOps, memoryRandomWeight)
}

func benchMemoryAlloc(duration time.Duration) result {
	ctx, cancel := context.WithTimeout(context.Background(), duration)
	defer cancel()

	var allocs uint64
	var checksum uint64
	var wg sync.WaitGroup
	workers := runtime.GOMAXPROCS(0)
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(seed int) {
			defer wg.Done()
			ring := make([][]byte, 1024)
			ringPos := 0
			localAllocs := uint64(0)
			localChecksum := uint64(0)
			for {
				select {
				case <-ctx.Done():
					atomic.AddUint64(&allocs, localAllocs)
					atomic.AddUint64(&checksum, localChecksum)
					return
				default:
					for j := 0; j < 256; j++ {
						b := make([]byte, 4096)
						b[seed%len(b)] = byte(j)
						localChecksum += uint64(b[seed%len(b)])
						ring[ringPos] = b
						ringPos = (ringPos + 1) % len(ring)
					}
					localAllocs += 256
				}
			}
		}(i)
	}
	wg.Wait()

	kallocs := float64(allocs) / duration.Seconds() / 1000
	return metricResult("MEM", "Memory alloc", fmt.Sprintf("%.2f K allocs/s", kallocs),
		fmt.Sprintf("4 KiB allocations, checksum %d", checksum), kallocs, baseMemoryAllocKAllocs, memoryAllocWeight)
}

func benchDisk(dir string, fileMB int, progress *progressTracker) []result {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return []result{failedResult("DISK", "DISK TOTAL", diskOverallWeight, err)}
	}
	path := filepath.Join(dir, fmt.Sprintf("vpsbench-%d.tmp", time.Now().UnixNano()))
	defer os.Remove(path)

	seqWrite, seqRead := benchDiskSequential(path, fileMB)
	progress.complete("DISK Disk seq write")
	progress.complete("DISK Disk seq read")
	random := benchDiskRandom(path)
	progress.complete("DISK Disk random rw")
	meta := benchDiskMetadata(dir)
	progress.complete("DISK Disk metadata")

	items := []result{seqWrite, seqRead, random, meta}
	total := weightedScore(items)
	for i := range items {
		items[i].Weight = 0
	}
	items = append(items, result{
		Group: "DISK", Name: "DISK TOTAL", Value: grade(total),
		Note: "weighted disk score", Score: total, Weight: diskOverallWeight,
	})
	return items
}

func benchDiskSequential(path string, fileMB int) (result, result) {
	size := int64(fileMB) * 1024 * 1024
	buf := make([]byte, 4*1024*1024)
	for i := range buf {
		buf[i] = byte(i*31 + 7)
	}

	start := time.Now()
	f, err := os.OpenFile(path, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0644)
	if err != nil {
		return failedResult("DISK", "Disk seq write", diskSeqWriteWeight, err), failedResult("DISK", "Disk seq read", diskSeqReadWeight, err)
	}
	written := int64(0)
	for written < size {
		n := int64(len(buf))
		if remain := size - written; remain < n {
			n = remain
		}
		if _, err := f.Write(buf[:n]); err != nil {
			_ = f.Close()
			return failedResult("DISK", "Disk seq write", diskSeqWriteWeight, err), failedResult("DISK", "Disk seq read", diskSeqReadWeight, err)
		}
		written += n
	}
	if err := f.Sync(); err != nil {
		_ = f.Close()
		return failedResult("DISK", "Disk seq write", diskSeqWriteWeight, err), failedResult("DISK", "Disk seq read", diskSeqReadWeight, err)
	}
	if err := f.Close(); err != nil {
		return failedResult("DISK", "Disk seq write", diskSeqWriteWeight, err), failedResult("DISK", "Disk seq read", diskSeqReadWeight, err)
	}
	writeMBps := float64(written) / time.Since(start).Seconds() / 1024 / 1024

	start = time.Now()
	f, err = os.Open(path)
	if err != nil {
		return metricResult("DISK", "Disk seq write", fmt.Sprintf("%.2f MB/s", writeMBps), fmt.Sprintf("%d MiB file", fileMB), writeMBps, baseDiskSeqWriteMBps, diskSeqWriteWeight),
			failedResult("DISK", "Disk seq read", diskSeqReadWeight, err)
	}
	read, err := io.Copy(io.Discard, f)
	closeErr := f.Close()
	if err != nil {
		return metricResult("DISK", "Disk seq write", fmt.Sprintf("%.2f MB/s", writeMBps), fmt.Sprintf("%d MiB file", fileMB), writeMBps, baseDiskSeqWriteMBps, diskSeqWriteWeight),
			failedResult("DISK", "Disk seq read", diskSeqReadWeight, err)
	}
	if closeErr != nil {
		return metricResult("DISK", "Disk seq write", fmt.Sprintf("%.2f MB/s", writeMBps), fmt.Sprintf("%d MiB file", fileMB), writeMBps, baseDiskSeqWriteMBps, diskSeqWriteWeight),
			failedResult("DISK", "Disk seq read", diskSeqReadWeight, closeErr)
	}
	readMBps := float64(read) / time.Since(start).Seconds() / 1024 / 1024

	return metricResult("DISK", "Disk seq write", fmt.Sprintf("%.2f MB/s", writeMBps), fmt.Sprintf("%d MiB file", fileMB), writeMBps, baseDiskSeqWriteMBps, diskSeqWriteWeight),
		metricResult("DISK", "Disk seq read", fmt.Sprintf("%.2f MB/s", readMBps), fmt.Sprintf("%d MiB file", fileMB), readMBps, baseDiskSeqReadMBps, diskSeqReadWeight)
}

func benchDiskRandom(path string) result {
	const fileSize = 64 * 1024 * 1024
	const blockSize = 4096
	const ops = 4096

	f, err := os.OpenFile(path+".rnd", os.O_CREATE|os.O_TRUNC|os.O_RDWR, 0644)
	if err != nil {
		return failedResult("DISK", "Disk random rw", diskRandomRWWeight, err)
	}
	defer os.Remove(path + ".rnd")
	defer f.Close()

	buf := make([]byte, blockSize)
	for i := range buf {
		buf[i] = byte(i * 19)
	}
	if err := f.Truncate(fileSize); err != nil {
		return failedResult("DISK", "Disk random rw", diskRandomRWWeight, err)
	}

	r := rand.New(rand.NewSource(42))
	start := time.Now()
	var checksum uint32
	for i := 0; i < ops; i++ {
		off := int64(r.Intn(fileSize/blockSize)) * blockSize
		if i%2 == 0 {
			if _, err := f.WriteAt(buf, off); err != nil {
				return failedResult("DISK", "Disk random rw", diskRandomRWWeight, err)
			}
		} else {
			if _, err := f.ReadAt(buf, off); err != nil {
				return failedResult("DISK", "Disk random rw", diskRandomRWWeight, err)
			}
			checksum += crc32.ChecksumIEEE(buf)
		}
	}
	_ = f.Sync()
	mbps := float64(ops*blockSize) / time.Since(start).Seconds() / 1024 / 1024
	return metricResult("DISK", "Disk random rw", fmt.Sprintf("%.2f MB/s", mbps),
		fmt.Sprintf("4 KiB mixed IO, checksum %d", checksum), mbps, baseDiskRandomRWMBps, diskRandomRWWeight)
}

func benchDiskMetadata(dir string) result {
	testDir := filepath.Join(dir, fmt.Sprintf("vpsbench-meta-%d", time.Now().UnixNano()))
	if err := os.MkdirAll(testDir, 0755); err != nil {
		return failedResult("DISK", "Disk metadata", diskMetadataWeight, err)
	}
	defer os.RemoveAll(testDir)

	const files = 1000
	start := time.Now()
	for i := 0; i < files; i++ {
		name := filepath.Join(testDir, fmt.Sprintf("f-%04d.tmp", i))
		if err := os.WriteFile(name, []byte("vpsbench"), 0644); err != nil {
			return failedResult("DISK", "Disk metadata", diskMetadataWeight, err)
		}
		if _, err := os.Stat(name); err != nil {
			return failedResult("DISK", "Disk metadata", diskMetadataWeight, err)
		}
		if err := os.Remove(name); err != nil {
			return failedResult("DISK", "Disk metadata", diskMetadataWeight, err)
		}
	}
	kops := float64(files*3) / time.Since(start).Seconds() / 1000
	return metricResult("DISK", "Disk metadata", fmt.Sprintf("%.2f K ops/s", kops),
		"create/stat/delete small files", kops, baseDiskMetadataKOpsSec, diskMetadataWeight)
}

func runParallelBytes(duration time.Duration, workers int, fn func() (int, int)) (float64, uint64) {
	ctx, cancel := context.WithTimeout(context.Background(), duration)
	defer cancel()

	var bytesDone uint64
	var checksum uint64
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			localBytes := uint64(0)
			localChecksum := uint64(0)
			for {
				select {
				case <-ctx.Done():
					atomic.AddUint64(&bytesDone, localBytes)
					atomic.AddUint64(&checksum, localChecksum)
					return
				default:
					n, c := fn()
					localBytes += uint64(n)
					localChecksum += uint64(c)
				}
			}
		}()
	}
	wg.Wait()
	mbps := float64(bytesDone) / duration.Seconds() / 1024 / 1024
	return mbps, checksum
}

func runParallelUnits(duration time.Duration, workers int, unitsPerRun int64, fn func() int) (float64, uint64) {
	ctx, cancel := context.WithTimeout(context.Background(), duration)
	defer cancel()

	var units uint64
	var checksum uint64
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			localUnits := uint64(0)
			localChecksum := uint64(0)
			for {
				select {
				case <-ctx.Done():
					atomic.AddUint64(&units, localUnits)
					atomic.AddUint64(&checksum, localChecksum)
					return
				default:
					localChecksum += uint64(fn())
					localUnits += uint64(unitsPerRun)
				}
			}
		}()
	}
	wg.Wait()
	return float64(units) / duration.Seconds(), checksum
}

func makeBinaryCorpus(size int) []byte {
	out := make([]byte, size)
	x := uint64(0x123456789abcdef)
	for i := range out {
		x ^= x << 13
		x ^= x >> 7
		x ^= x << 17
		if i%97 < 17 {
			out[i] = byte("package benchmark data "[i%23])
		} else {
			out[i] = byte(x)
		}
	}
	return out
}

func makeMarkdownCorpus(size int) string {
	var b strings.Builder
	for b.Len() < size {
		b.WriteString("# Benchmark Notes\n")
		b.WriteString("This paragraph links to [docs](https://example.com) and contains `inline code`.\n")
		b.WriteString("## Results\n- compression\n- image processing\n- machine learning\n\n")
	}
	s := b.String()
	return s[:size]
}

func detectMemoryGB(override float64) (float64, string) {
	if override > 0 {
		return override, "manual override"
	}
	switch runtime.GOOS {
	case "linux":
		if b, err := os.ReadFile("/proc/meminfo"); err == nil {
			for _, line := range strings.Split(string(b), "\n") {
				if strings.HasPrefix(line, "MemTotal:") {
					fields := strings.Fields(line)
					if len(fields) >= 2 {
						kib, err := strconv.ParseFloat(fields[1], 64)
						if err == nil {
							return kib / 1024 / 1024, "/proc/meminfo"
						}
					}
				}
			}
		}
	case "darwin":
		if out, err := exec.Command("sysctl", "-n", "hw.memsize").Output(); err == nil {
			bytes, err := strconv.ParseFloat(strings.TrimSpace(string(out)), 64)
			if err == nil {
				return bytes / 1024 / 1024 / 1024, "sysctl hw.memsize"
			}
		}
	case "windows":
		if out, err := exec.Command("wmic", "computersystem", "get", "TotalPhysicalMemory", "/value").Output(); err == nil {
			for _, line := range strings.Split(string(out), "\n") {
				if strings.HasPrefix(strings.TrimSpace(line), "TotalPhysicalMemory=") {
					value := strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(line), "TotalPhysicalMemory="))
					bytes, err := strconv.ParseFloat(value, 64)
					if err == nil {
						return bytes / 1024 / 1024 / 1024, "wmic TotalPhysicalMemory"
					}
				}
			}
		}
	}
	return 1.0, "fallback; use -mem-gb to override"
}

func resultFilename(t time.Time) string {
	base := t.Format("200601021504")
	name := base + "result.txt"
	if _, err := os.Stat(name); os.IsNotExist(err) {
		return name
	}
	for i := 2; ; i++ {
		name = fmt.Sprintf("%s_%dresult.txt", base, i)
		if _, err := os.Stat(name); os.IsNotExist(err) {
			return name
		}
	}
}

func metricResult(group, name, value, note string, metric, baseline, weight float64) result {
	return result{
		Group:  group,
		Name:   name,
		Value:  value,
		Note:   note,
		Score:  metricScore(metric, baseline),
		Weight: weight,
	}
}

func failedResult(group, name string, weight float64, err error) result {
	return result{Group: group, Name: name, Value: "failed", Note: err.Error(), Weight: weight}
}

func metricScore(value, baseline float64) float64 {
	if value <= 0 || baseline <= 0 {
		return 0
	}
	return value / baseline * baseScore
}

func weightedScore(results []result) float64 {
	return totalScoreOnly(results)
}

func totalScoreOnly(results []result) float64 {
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

func renderReport(header string, results []result) string {
	var b strings.Builder
	b.WriteString(header)
	b.WriteString("\n")
	b.WriteString(fmt.Sprintf("%-8s %-20s %-24s %-9s %s\n", "Group", "Test", "Result", "Score", "Note"))
	b.WriteString(strings.Repeat("-", 112) + "\n")
	for _, r := range results {
		b.WriteString(fmt.Sprintf("%-8s %-20s %-24s %-9.0f %s\n", r.Group, r.Name, r.Value, r.Score, r.Note))
	}

	b.WriteString(strings.Repeat("-", 112) + "\n")
	total, weight := totalScore(results)
	b.WriteString(fmt.Sprintf("%-8s %-20s %-24s %-9.0f %s\n", "ALL", "OVERALL TOTAL", grade(total), total, fmt.Sprintf("weighted score, %.0f active weight", weight)))
	b.WriteString("\n")
	b.WriteString("Baselines: 1000 points ~= small 1 vCPU / 1 GiB VPS reference. Scores are not Geekbench-compatible.\n")
	b.WriteString(fmt.Sprintf("CPU: gzip %.0f MB/s, text %.0f MB/s, image %.0f MPix/s, ML %.1f GFLOPS, ray %.0f MRays/s, SHA1 %.0f MB/s\n",
		baseCPUCompressionMBps, baseCPUTextMBps, baseCPUImageMPix, baseCPUMLGFLOPS, baseCPURenderMRays, baseCPUCryptoMBps))
	b.WriteString(fmt.Sprintf("MEM per-GB: copy %.1f GB/s, random %.0f M ops/s, alloc %.0f K/s. MEMORY TOTAL = per-GB score * sqrt(total GiB).\n",
		baseMemoryCopyGBps, baseMemoryRandomMOps, baseMemoryAllocKAllocs))
	b.WriteString(fmt.Sprintf("DISK: write %.0f MB/s, read %.0f MB/s, random %.0f MB/s, meta %.0f K ops/s\n",
		baseDiskSeqWriteMBps, baseDiskSeqReadMBps, baseDiskRandomRWMBps, baseDiskMetadataKOpsSec))
	return b.String()
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
