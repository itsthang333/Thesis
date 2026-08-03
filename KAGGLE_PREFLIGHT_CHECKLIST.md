# Kaggle preflight checklist cho thesis

File này là checklist sống để chặn các lỗi Kaggle/transport/runtime đã lặp lại
trong project. Phải đọc toàn bộ và ghi bằng chứng PASS/FAIL vào prelaunch audit
trước **mỗi** `kaggle kernels push`. Không đánh dấu theo trí nhớ; mỗi ô phải có
command, hash, test hoặc artifact kiểm chứng.

Khi gặp một lỗi có khả năng tái diễn:

1. Ghi error boundary và failure analysis vào `RESEARCH_LOG.md`, commit/push.
2. Thêm lỗi vào bảng catalog bên dưới trước khi sửa hoặc rerun.
3. Thêm guard/test fail-closed tương ứng vào wrapper/binder/auditor.
4. Chỉ rerun sau khi checklist, correction và prelaunch audit đã visible trên
   branch điều phối.

## Checklist bắt buộc trước push

### 1. Điều phối và Git

- [ ] Fetch `origin/research-wsss-improvement` và nhánh cộng tác; đọc toàn bộ
  `AGENTS.md`, `RESEARCH_LOG.md` và file này.
- [ ] Claim thực nghiệm là duy nhất, đang `ĐANG LÀM`, đã commit/push central và
  không trùng claim khác.
- [ ] `git status --short --branch` không có thay đổi ngoài package dự kiến; ghi
  rõ HEAD, origin HEAD và scientific-source commit.
- [ ] Không force-push, không đóng gói worktree chưa commit, không dùng source
  commit được đoán từ short SHA.

### 2. Source, protocol và line ending

- [ ] Protocol/addendum/readiness khóa exact full 64-char SHA-256 của source,
  split, model, input, freeze và output consumer cần thiết.
- [ ] Hash source được tính trên canonical Git-LF bytes; nếu input lịch sử là
  CRLF thì wrapper phải dựng CRLF một lần từ Git-LF và kiểm tra **cả hai** hash.
- [ ] Không giả định `git checkout`, `checkout-index` hoặc raw `git archive` sẽ
  tạo LF: tracked blob tự nó có thể là CRLF. Trước hết hash raw blob; nếu contract
  khóa canonical-LF thì dùng một normalizer fail-closed từ verified raw bytes,
  rồi assert cả input SHA và output canonical-LF SHA trước consumer.
- [ ] Binder chỉ thay đúng các launch fields đã khai báo, rồi inverse-reconstruct
  byte-for-byte về template; packaged source ngoài binding phải giống checkout.
- [ ] `py_compile`, focused tests, metadata JSON parse, protocol closure,
  inverse-binding và `git diff --check` đều PASS trên exact package.
- [ ] Seed, bootstrap count, arm order, threshold, cohort và evaluator/decision
  hash khớp giữa protocol, wrapper và comparator; không dùng default ngầm.

PowerShell canonical-LF hash mẫu:

```powershell
$text = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $path))
$text = $text.Replace("`r`n", "`n").Replace("`r", "`n")
$bytes = [Text.UTF8Encoding]::new($false).GetBytes($text)
$sha = [Security.Cryptography.SHA256]::Create()
([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant()
```

### 3. Runtime và dependency

- [ ] In `sys.executable`, `sys.version` và exact version của mọi dependency
  mutable trước real input; cài/pin version trước tests nếu protocol yêu cầu.
- [ ] Chạy focused tests bằng **chính interpreter/runtime sẽ thực thi**. Không
  suy từ `py_compile` rằng NumPy/PyTorch/sklearn/transformers đã có.
- [ ] Audit mọi API phụ thuộc phiên bản, đặc biệt `zip(..., strict=True)` cần
  Python >=3.10. Nếu hỗ trợ Python 3.9, dùng explicit length check + ordinary
  `zip` và regression test tương đương.
- [ ] Không nhầm interpreter: môi trường model có thể không có Kaggle CLI;
  system interpreter có Kaggle CLI có thể không có model dependencies.
- [ ] Static/synthetic tests chạy trước download model lớn, resolve BTXRD input,
  training hoặc inference.

### 4. Kaggle metadata và transport

- [ ] `kernel-metadata.json` là JSON hợp lệ; slug/title/code filename không vượt
  giới hạn API đã audit; kernel để private.
- [ ] Accelerator khai đúng T4x2/P100. Wrapper kiểm tra real CUDA convolution
  trên đúng số GPU, không chỉ `torch.cuda.is_available()`.
- [ ] `dataset_sources` và `kernel_sources` đúng loại transport. Output của một
  kernel terminal `ERROR` không được giả định là attachable kernel source;
  đóng gói immutable bytes thành private dataset, kèm inventory/archive hash.
- [ ] Wrapper xử lý server-expanded/double-root layout bằng exact hash/count,
  không chọn thư mục đầu tiên hoặc dựa tên gần đúng.
- [ ] Input archive/dataset không chứa GT/test/evaluator-only table ngoài boundary
  được phép; reject path/key khả nghi (`mask`, `polygon`, `ground_truth`, `test`,
  object array) theo contract của experiment.
- [ ] Version, checkout, wrapper, metadata và launch-binding đều đã freeze; package
  có `LAUNCH_BINDING_READY=True` và version đúng lần push sắp tới.

### 5. Device và numerical reproducibility

- [ ] Producer và independent auditor chạy phép tính nhạy cảm trên cùng device
  đã khóa. Không tự chọn CUDA chỉ vì GPU hiện diện nếu producer đã chuyển CPU.
- [ ] Original/flip, spatial-null, rank, quantile, float16 serialization và tie
  arithmetic có regression trên device/runtime đích.
- [ ] Tolerance dựa trên ULP/epsilon và magnitude thực tế; test cả phía trong và
  ngoài tolerance. Không đặt absolute `1e-6` tùy ý cho reduction float32.
- [ ] Auditor so serialized evidence theo serialization-aware bound; không đòi
  tái lập float64 trước-serialization từ payload float16/float32.

### 6. Data boundary và safety

- [ ] Prediction pair và mọi score/map manifest đã freeze vật lý trước GT.
- [ ] Loader validation không nhận full split nếu nó verify mọi hàng trước khi
  lọc: phải dùng projection chỉ gồm validation để không đọc train/test bytes.
- [ ] Chứng minh projection chỉ có `split=val`, đúng cohort và ghi projection SHA
  trong evaluation audit.
- [ ] Không validation polygon/GT trong training, selector fit hoặc pre-GT audit;
  không evaluator-only per-image table trước readiness.
- [ ] Không train consumer trước operational pass; BTXRD test luôn khóa; không
  truy cập output Kaggle của cộng tác viên nếu scope chỉ cho phép đọc Git log.

### 7. Output, status và download

- [ ] Output directory mới, không tồn tại; wrapper fail-closed nếu output cũ có
  thể bị trộn.
- [ ] Output inventory ghi exact count/bytes/hash và không có `.part`.
- [ ] Không gọi status ngay sau push, không polling liên tục, không tạo monitor
  nếu người dùng chưa yêu cầu. Mỗi nhịp chỉ một bounded status check.
- [ ] Khi terminal, dùng `project/download_kaggle_output_inventory.py` vào temp
  ignored mới; tải compact audit/log trước. Bulk output timeout là transport
  error, resume atomic theo inventory, không kết luận kernel fail.
- [ ] Direct log, run manifest, pair freeze, audit/decision và mọi metric file có
  SHA-256; error cũng phải giữ evidence và cập nhật log trước correction/rerun.

## Catalog lỗi tái diễn và guard bắt buộc

| ID | Triệu chứng/nguồn lỗi | Guard phải có trước push |
|---|---|---|
| KPF-001 | Split/source hash lệch do canonical-LF và tracked/local/Kaggle CRLF; ngay cả raw Git blob có thể là CRLF | Khóa raw + canonical-LF SHA; lấy raw blob và rehash; chạy fail-closed CRLF/LF normalizer đúng một lần; assert input/output; không suy line ending từ tên thao tác Git. |
| KPF-002 | Kaggle base image đổi dependency (`transformers 5.x` thay vì `4.50.2`) | Pin exact version, import-assert version, chạy trước model download/real input. |
| KPF-003 | Python 3.9 lỗi `zip(..., strict=True)` | In runtime version; native test trên runtime đích; explicit length check + ordinary `zip` nếu cần 3.9. |
| KPF-004 | Chạy nhầm interpreter: thiếu NumPy/PyTorch/sklearn hoặc thiếu `kaggle` | In `sys.executable`; probe imports; tách rõ interpreter scientific và interpreter Kaggle CLI. |
| KPF-005 | Package bind từ checkout/source khác hoặc launch field chưa khóa | Exact ancestry + full SHA closure + three-field inverse reconstruction + clean checkout. |
| KPF-006 | Kaggle `SaveKernel` HTTP 400 do slug/title quá dài | Static length guard cho slug/title/filename và exact metadata test trước API call. |
| KPF-007 | Terminal-ERROR kernel bị Kaggle từ chối làm `kernel_source` | Không attach trực tiếp; đóng immutable output thành private dataset, inventory/archive hash, `kernel_sources=[]`. |
| KPF-008 | Producer CPU nhưng auditor tự chọn CUDA, null/tie arithmetic lệch | Device là provenance field; force cùng device; monkeypatch regression khi CUDA available. |
| KPF-009 | Absolute tolerance quá chặt cho float32 original/flip reduction | Dùng bound theo ULP + epsilon floor; test numerical boundary trên T4x2 runtime. |
| KPF-010 | Test tolerance tự mâu thuẫn với epsilon floor | Test separate ULP-dominated và epsilon-dominated regimes; kiểm tra cả accept/reject. |
| KPF-011 | Kaggle input bị server-expand/double-root, locator không tìm thấy artifact | Recursive exact locator bằng unique archive/hash/count; test layout server-expanded. |
| KPF-012 | Bulk output download timeout hoặc file dở | Inventory downloader atomic/resumable, compact-first, reject `.part`; không coi là scientific fail. |
| KPF-013 | Audit serialized float16/float32 bằng exact pre-serialization value | Recompute cùng serialization hoặc dùng proven serialization-aware bound; khóa dtype. |
| KPF-014 | Segmentation loader nhận full manifest và đọc test trước khi lọc val | Hash full manifest GT-blind, tạo val-only projection, assert 371 val rows/0 test và audit projection SHA. |
| KPF-015 | Protocol seed/default không khớp comparator | Explicit seed/count/order trong protocol, CLI và launch binding; source test so exact values. |
| KPF-016 | GPU metadata đúng nhưng job chỉ dùng một GPU hoặc không có real CUDA | Two-device real-convolution guard và logged device names trước scientific input. |
| KPF-017 | Status/output bị kiểm tra quá sớm hoặc lặp | Không immediate poll; một bounded check mỗi nhịp; terminal mới tải direct log/inventory. |
| KPF-018 | Output cũ trộn với rerun | Unique ignored output root; `exist_ok=False`; audit requested path và file inventory. |

## Mẫu evidence tối thiểu trong prelaunch audit

Mỗi audit phải ghi: experiment ID, owner, UTC/ICT time, branch/HEAD/origin,
scientific commit, execution checkout, protocol/addendum SHA, wrapper/metadata/
binder SHA, canonical-LF source closure, target Python/dependency versions,
focused-test counts, GPU guard, exact input hashes/counts, output contract,
prediction/GT/consumer/test locks, từng KPF ID liên quan và `authorized_launch`.
