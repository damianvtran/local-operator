# vi mobile.doctor audit a6, PR #4216 round 2

Verdict: PASSED (0 blocker / 0 major / 0 minor / 0 nit).

Scope: 4 keys changed by 193e08e33. English source sha256 8e22fc91780c70e0750c05c92a8478db0ca00d738981ae20063028fe33b0ddcc.

Select arms complete; 'handle theo dõi' and 'lượt theo dõi tệp' terminology matches round 1; 'Phần lớn do … giữ' natural.

## Renders (synthetic params: node, pid 4242, systemctl --user restart dev.service)

- `near` / `command`: Tài khoản của bạn đang dùng 8100/8192 lượt theo dõi tệp được phép trên máy tính này, gần chạm giới hạn. Phần lớn thuộc về node (tiến trình 4242): hãy khởi động lại nó bằng systemctl --user restart dev.service, hoặc nhờ quản trị viên tăng fs.inotify.max_user_watches.
- `near` / `program`: Tài khoản của bạn đang dùng 8100/8192 lượt theo dõi tệp được phép trên máy tính này, gần chạm giới hạn. Phần lớn thuộc về node (tiến trình 4242): hãy khởi động lại chương trình đó, hoặc nhờ quản trị viên tăng fs.inotify.max_user_watches.
- `near` / `other`: Tài khoản của bạn đang dùng 8100/8192 lượt theo dõi tệp được phép trên máy tính này, gần chạm giới hạn. Hãy nhờ quản trị viên tăng fs.inotify.max_user_watches.
- `at_limit` / `command`: Tài khoản của bạn đã dùng hết lượt theo dõi tệp trên máy tính này (8100/8192), nên các chương trình theo dõi tệp sẽ báo lỗi “No space left on device”. Phần lớn thuộc về node (tiến trình 4242): hãy khởi động lại nó bằng systemctl --user restart dev.service, hoặc nhờ quản trị viên tăng fs.inotify.max_user_watches.
- `at_limit` / `program`: Tài khoản của bạn đã dùng hết lượt theo dõi tệp trên máy tính này (8100/8192), nên các chương trình theo dõi tệp sẽ báo lỗi “No space left on device”. Phần lớn thuộc về node (tiến trình 4242): hãy khởi động lại chương trình đó, hoặc nhờ quản trị viên tăng fs.inotify.max_user_watches.
- `at_limit` / `other`: Tài khoản của bạn đã dùng hết lượt theo dõi tệp trên máy tính này (8100/8192), nên các chương trình theo dõi tệp sẽ báo lỗi “No space left on device”. Hãy nhờ quản trị viên tăng fs.inotify.max_user_watches.
- `instances_near` / `command`: Tài khoản của bạn đang dùng 126/128 handle theo dõi được phép trên máy tính này, gần chạm giới hạn. Phần lớn do node (tiến trình 4242) giữ: hãy khởi động lại nó bằng systemctl --user restart dev.service, hoặc nhờ quản trị viên tăng fs.inotify.max_user_instances.
- `instances_near` / `program`: Tài khoản của bạn đang dùng 126/128 handle theo dõi được phép trên máy tính này, gần chạm giới hạn. Phần lớn do node (tiến trình 4242) giữ: hãy khởi động lại chương trình đó, hoặc nhờ quản trị viên tăng fs.inotify.max_user_instances.
- `instances_near` / `other`: Tài khoản của bạn đang dùng 126/128 handle theo dõi được phép trên máy tính này, gần chạm giới hạn. Hãy nhờ quản trị viên tăng fs.inotify.max_user_instances.
- `instances_at_limit` / `command`: Tài khoản của bạn đã dùng hết handle theo dõi trên máy tính này (126/128), nên các chương trình theo dõi tệp sẽ báo lỗi “Too many open files”. Phần lớn do node (tiến trình 4242) giữ: hãy khởi động lại nó bằng systemctl --user restart dev.service, hoặc nhờ quản trị viên tăng fs.inotify.max_user_instances.
- `instances_at_limit` / `program`: Tài khoản của bạn đã dùng hết handle theo dõi trên máy tính này (126/128), nên các chương trình theo dõi tệp sẽ báo lỗi “Too many open files”. Phần lớn do node (tiến trình 4242) giữ: hãy khởi động lại chương trình đó, hoặc nhờ quản trị viên tăng fs.inotify.max_user_instances.
- `instances_at_limit` / `other`: Tài khoản của bạn đã dùng hết handle theo dõi trên máy tính này (126/128), nên các chương trình theo dõi tệp sẽ báo lỗi “Too many open files”. Hãy nhờ quản trị viên tăng fs.inotify.max_user_instances.
