# Quy tắc phối hợp nghiên cứu thesis

File này là chỉ dẫn bắt buộc cho mọi người và mọi Codex task/chat làm việc trong
repository này. Khi bắt đầu hoặc tiếp quản một task mới, phải đọc toàn bộ file
này và toàn bộ `RESEARCH_LOG.md` trước khi thực hiện nghiên cứu.

## 1. Mục tiêu và nguồn điều phối chung

- Tất cả thành viên cùng hướng tới mục tiêu thesis và goal nghiên cứu hiện tại.
- `RESEARCH_LOG.md` là nguồn điều phối thực nghiệm trung tâm và có thẩm quyền.
- Branch `research-wsss-improvement` giữ bản `RESEARCH_LOG.md` điều phối chính.
  Người cộng tác làm code trên branch riêng và đồng bộ bản log mới nhất từ branch
  này trước khi đăng ký hoặc chạy thực nghiệm.
- Không dựa riêng vào lịch sử chat, trí nhớ hoặc file tạm để xác định thực nghiệm
  nào đang chạy hay đã hoàn thành.
- Không xóa, ghi đè hoặc sửa lại kết quả/bằng chứng của thành viên khác. Nếu cần
  hiệu chính, thêm một mục mới có liên kết rõ đến mục cũ và giải thích lý do.

## 2. Quy trình bắt buộc trước mọi thực nghiệm

Trước khi chạy bất kỳ thực nghiệm khoa học nào:

1. Fetch phiên bản mới nhất của `origin/research-wsss-improvement`.
2. Nếu đang làm trên branch riêng, đồng bộ phiên bản `RESEARCH_LOG.md` mới nhất
   từ `origin/research-wsss-improvement` vào branch đó mà không ghi đè code hoặc
   thay đổi cục bộ.
3. Đọc **toàn bộ** `RESEARCH_LOG.md`, không chỉ tail hoặc mục gần nhất.
4. Tìm các thực nghiệm giống, gần giống, có chung input, hypothesis, protocol,
   model, candidate gallery, evaluator hoặc output consumer.
5. Không chạy lại một thực nghiệm đang có trạng thái `ĐANG LÀM`.
6. Nếu chưa có người phụ trách, thêm một mục đăng ký vào `RESEARCH_LOG.md` với
   trạng thái `ĐANG LÀM`.
7. Commit và push ghi chú `ĐANG LÀM` trước khi chạy. Không còn yêu cầu commit này
   chỉ chứa `RESEARCH_LOG.md`; code có thể được commit/push bình thường theo quy
   trình của branch đang làm việc.
8. Bảo đảm ghi chú `ĐANG LÀM` đã hiện diện trong `RESEARCH_LOG.md` điều phối trên
   `research-wsss-improvement` để thành viên ở branch khác nhìn thấy. Người làm
   trên branch riêng có thể đưa thay đổi log về branch điều phối bằng một commit,
   merge hoặc PR chỉ dành cho phần log.
9. Chỉ bắt đầu chạy sau khi đăng ký đã được push thành công và nhìn thấy từ
   branch điều phối mới nhất.

Mỗi thành viên push code lên branch của mình như bình thường; việc push code là
cần thiết để kiểm tra source, chạy Kaggle và đóng băng provenance. Không hạn chế
push code chỉ vì cơ chế điều phối log. Tuy nhiên, không push code của branch này
đè lên branch code của người khác và không force-push.

Khi chỉ cần lấy log điều phối, không dùng một thao tác có thể merge/ghi đè code
ngoài ý muốn. Việc đồng bộ riêng `RESEARCH_LOG.md` phải bảo toàn worktree và code
trên branch hiện tại. Nếu log có xung đột, fetch lại và hợp nhất đầy đủ nội dung;
không bỏ mục hoặc kết quả của người khác.

## 3. Nội dung tối thiểu của một đăng ký thực nghiệm

Mỗi mục `ĐANG LÀM` phải có đủ:

- Mã thực nghiệm duy nhất, ưu tiên dạng
  `EXP-YYYYMMDD-<owner>-<short-slug>`.
- Người phụ trách.
- Thời điểm đăng ký và commit đăng ký.
- Trạng thái `ĐANG LÀM`.
- Mục tiêu/hypothesis cụ thể.
- Phạm vi thay đổi và điểm khác với các thực nghiệm gần nhất.
- Mã các thực nghiệm được tham khảo hoặc kế thừa.
- Input, candidate gallery, split, protocol và artifact/hash quan trọng.
- Loại compute dự kiến và nơi chạy.
- Output/gate dự kiến dùng để kết luận.
- Các khóa an toàn liên quan: prediction freeze, validation GT, consumer và test.

Hai thực nghiệm không được dùng mã trùng nhau. Nếu scope thay đổi đáng kể sau khi
đăng ký, phải cập nhật log và push trước khi chạy scope mới.

## 4. Kế thừa và tránh trùng lặp

- Được phép tham khảo và kế thừa mọi kết quả đã có.
- Khi kế thừa, phải ghi rõ mã thực nghiệm nguồn và exact artifact/hash được dùng.
- Việc đọc và đồng bộ `RESEARCH_LOG.md` của nhánh cộng tác không chỉ để tránh
  trùng lặp. Phải chủ động rút ra kỹ thuật, bằng chứng, kết quả dương/âm, error
  boundary và insight có thể chuyển giao; dùng chúng để cải thiện giả thuyết kế
  nhiệm hoặc thu hẹp không gian thử nghiệm chung.
- Chỉ kế thừa một kỹ thuật vào mô hình/hướng cải tiến vì lý do hiệu năng khi nó
  đã có kết quả terminal, được audit đúng prediction-freeze/GT boundary và tốt
  hơn baseline liên quan theo metric/gate chung đã định trước. Code mới, protocol,
  oracle ceiling, kết quả đang chạy hoặc kỳ vọng lý thuyết chưa đủ để kết luận
  kỹ thuật đó tốt hơn. Kết quả âm/error vẫn được học để tránh lặp lại hoặc thiết
  kế đối chứng, nhưng không được quảng bá như một cải tiến đã chứng minh.
- Mục tiêu là hai workstream cùng tiến trên một chuỗi bằng chứng. Không chạy lại
  nguyên implementation của thành viên khác; ưu tiên kế thừa phần đã chứng minh
  hữu ích và đăng ký một thay đổi khoa học chưa được chạy, có giá trị thông tin
  mới. Nếu kết quả đang chờ, có thể chuẩn bị tĩnh một hướng kế nhiệm khác biệt
  nhưng không launch cạnh tranh.
- Khi một insight được chuyển từ workstream này sang workstream khác, ghi rõ mã
  thực nghiệm nguồn, commit/artifact/hash, phần được giữ lại, phần được thay đổi
  và vì sao thay đổi đó không phải bản sao của thực nghiệm nguồn.
- Không lặp lại thực nghiệm chỉ để “xác nhận lại” nếu không đăng ký một lý do mới,
  tiêu chí sai khác và giá trị thông tin dự kiến.
- Nếu một thực nghiệm liên quan đang `ĐANG LÀM`, không chạy cạnh tranh. Có thể
  chuẩn bị code/test tĩnh không tiêu thụ input khoa học, nhưng không launch,
  không tạo prediction và không đọc metric cho đến khi quyền thực nghiệm rõ ràng.
- Nếu phát hiện trùng sau khi đã đăng ký, dừng trước compute, cập nhật log và chọn
  một scope không trùng hoặc kế thừa kết quả của người đang phụ trách.

## 5. Cập nhật khi kết thúc hoặc gặp lỗi

Khi thực nghiệm kết thúc, cập nhật chính mục/mã thực nghiệm trong
`RESEARCH_LOG.md` và push với:

- Trạng thái cuối: `HOÀN THÀNH`, `LỖI`, `HỦY` hoặc `TẠM DỪNG`.
- Kernel/job/version, commit, protocol và exact SHA-256 của output/log chính.
- Kết quả và metric đầy đủ, kể cả kết quả âm hoặc chưa đạt goal.
- Error boundary/root cause nếu thất bại.
- Kết luận khoa học trung thực.
- Bước tiếp theo được đề xuất và mã thực nghiệm kế thừa nếu có.
- Xác nhận các khóa GT/consumer/test và prediction-freeze có được giữ hay không.

Kết quả cuối cũng phải được đưa về bản `RESEARCH_LOG.md` trên
`research-wsss-improvement`; chỉ lưu trên branch riêng thì chưa đủ để điều phối
và có thể khiến thành viên khác lặp lại thực nghiệm.

Không được xóa dấu vết của lỗi, kết quả âm hoặc thực nghiệm bị thay thế. Ghi một
mục kế nhiệm/supersession thay vì sửa lịch sử theo hướng làm mất bằng chứng.

## 6. Phân biệt chuẩn bị kỹ thuật và thực nghiệm

Các việc sau chưa được xem là một thực nghiệm khoa học nếu không mở dữ liệu khoa
học hoặc sinh/đọc kết quả khoa học:

- Viết code, unit test và synthetic test.
- Static audit, `py_compile`, lint hoặc test hồi quy trên dữ liệu tổng hợp.
- Chuẩn bị wrapper fail-closed chưa launch.
- Kiểm tra hash/provenance của artifact đã đóng băng.

Tuy nhiên, trước khi một công việc chuẩn bị chuyển sang launch, training,
inference trên cohort thật, prediction, evaluation hoặc đọc metric, phải đăng ký
`ĐANG LÀM` và push log theo quy trình ở trên.

## 7. Các ràng buộc nghiên cứu WSSS BTXRD hiện hành

- Huấn luyện WSSS chỉ dùng nhãn cấp ảnh.
- Mọi prediction validation phải được freeze vật lý trước khi mở validation GT.
- Không train consumer trước khi toàn bộ operational gate tương ứng pass.
- Không leakage, không tối ưu trực tiếp theo validation GT và không gian lận
  metric.
- BTXRD test luôn khóa cho đến khi goal/protocol chung cho phép rõ ràng.
- Compute nặng chỉ chạy trên Kaggle T4x2 hoặc P100 theo protocol đã đăng ký.
- Không polling Kaggle liên tục và không tạo monitor nếu người dùng chưa yêu cầu.
- Mọi kỹ thuật, paper, URL, protocol, experiment, error và kết luận phải được ghi
  đầy đủ vào `RESEARCH_LOG.md`.

## 8. Checklist khi chuyển task/chat

Một task/chat kế nhiệm phải:

1. Đọc toàn bộ `AGENTS.md` này.
2. Đọc toàn bộ `RESEARCH_LOG.md` mới nhất.
3. Kiểm tra branch, HEAD, origin và worktree; bảo toàn thay đổi chưa commit. Nếu
   đang ở branch cộng tác riêng, xác nhận log đã được đồng bộ từ
   `origin/research-wsss-improvement`.
4. Xác định các mã `ĐANG LÀM` và không chiếm/chạy trùng.
5. Tiếp tục từ artifact/protocol/hash mới nhất, không khởi động lại từ đầu.
6. Không coi mô tả handoff hoặc lịch sử chat là mạnh hơn trạng thái repository và
   artifact thực tế.
