# ur mobile.doctor audit a5, PR #4216 round 2

Verdict: PASSED (0 blocker / 0 major / 0 minor / 0 nit).

Scope: 4 keys changed by 193e08e33. English source sha256 8e22fc91780c70e0750c05c92a8478db0ca00d738981ae20063028fe33b0ddcc.

Select arms complete; logical order correct for RTL (Arabic-script clauses, LTR runs {name}/{pid}/{command}/sysctl embedded as in round 1, no stray bidi controls, ۔ full stop, ، comma); نگرانی ہینڈل / فائل نگرانیاں terminology matches round 1.

## Renders (synthetic params: node, pid 4242, systemctl --user restart dev.service)

- `near` / `command`: آپ کا اکاؤنٹ اس کمپیوٹر پر 8100 فائل نگرانیاں (حد 8192) استعمال کر رہا ہے، جو حد کے قریب ہے۔ ان میں سے زیادہ تر node (پروسیس 4242) کی ہیں: اسے systemctl --user restart dev.service سے دوبارہ شروع کریں، یا کسی منتظم سے fs.inotify.max_user_watches بڑھانے کو کہیں۔
- `near` / `program`: آپ کا اکاؤنٹ اس کمپیوٹر پر 8100 فائل نگرانیاں (حد 8192) استعمال کر رہا ہے، جو حد کے قریب ہے۔ ان میں سے زیادہ تر node (پروسیس 4242) کی ہیں: اس پروگرام کو دوبارہ شروع کریں، یا کسی منتظم سے fs.inotify.max_user_watches بڑھانے کو کہیں۔
- `near` / `other`: آپ کا اکاؤنٹ اس کمپیوٹر پر 8100 فائل نگرانیاں (حد 8192) استعمال کر رہا ہے، جو حد کے قریب ہے۔ کسی منتظم سے fs.inotify.max_user_watches بڑھانے کو کہیں۔
- `at_limit` / `command`: آپ کے اکاؤنٹ نے اس کمپیوٹر پر اپنی فائل نگرانیاں ختم کر لی ہیں (8100، حد 8192)، اس لیے فائلوں کی نگرانی کرنے والے پروگرام “No space left on device” کے ساتھ ناکام ہو جاتے ہیں۔ ان میں سے زیادہ تر node (پروسیس 4242) کی ہیں: اسے systemctl --user restart dev.service سے دوبارہ شروع کریں، یا کسی منتظم سے fs.inotify.max_user_watches بڑھانے کو کہیں۔
- `at_limit` / `program`: آپ کے اکاؤنٹ نے اس کمپیوٹر پر اپنی فائل نگرانیاں ختم کر لی ہیں (8100، حد 8192)، اس لیے فائلوں کی نگرانی کرنے والے پروگرام “No space left on device” کے ساتھ ناکام ہو جاتے ہیں۔ ان میں سے زیادہ تر node (پروسیس 4242) کی ہیں: اس پروگرام کو دوبارہ شروع کریں، یا کسی منتظم سے fs.inotify.max_user_watches بڑھانے کو کہیں۔
- `at_limit` / `other`: آپ کے اکاؤنٹ نے اس کمپیوٹر پر اپنی فائل نگرانیاں ختم کر لی ہیں (8100، حد 8192)، اس لیے فائلوں کی نگرانی کرنے والے پروگرام “No space left on device” کے ساتھ ناکام ہو جاتے ہیں۔ کسی منتظم سے fs.inotify.max_user_watches بڑھانے کو کہیں۔
- `instances_near` / `command`: آپ کا اکاؤنٹ اس کمپیوٹر پر 126 نگرانی ہینڈل (حد 128) استعمال کر رہا ہے، جو حد کے قریب ہے۔ سب سے زیادہ ہینڈل node (پروسیس 4242) کے پاس ہیں: اسے systemctl --user restart dev.service سے دوبارہ شروع کریں، یا کسی منتظم سے fs.inotify.max_user_instances بڑھانے کو کہیں۔
- `instances_near` / `program`: آپ کا اکاؤنٹ اس کمپیوٹر پر 126 نگرانی ہینڈل (حد 128) استعمال کر رہا ہے، جو حد کے قریب ہے۔ سب سے زیادہ ہینڈل node (پروسیس 4242) کے پاس ہیں: اس پروگرام کو دوبارہ شروع کریں، یا کسی منتظم سے fs.inotify.max_user_instances بڑھانے کو کہیں۔
- `instances_near` / `other`: آپ کا اکاؤنٹ اس کمپیوٹر پر 126 نگرانی ہینڈل (حد 128) استعمال کر رہا ہے، جو حد کے قریب ہے۔ کسی منتظم سے fs.inotify.max_user_instances بڑھانے کو کہیں۔
- `instances_at_limit` / `command`: آپ کے اکاؤنٹ نے اس کمپیوٹر پر اپنے نگرانی ہینڈل ختم کر لیے ہیں (126، حد 128)، اس لیے فائلوں کی نگرانی کرنے والے پروگرام “Too many open files” کے ساتھ ناکام ہو جاتے ہیں۔ سب سے زیادہ ہینڈل node (پروسیس 4242) کے پاس ہیں: اسے systemctl --user restart dev.service سے دوبارہ شروع کریں، یا کسی منتظم سے fs.inotify.max_user_instances بڑھانے کو کہیں۔
- `instances_at_limit` / `program`: آپ کے اکاؤنٹ نے اس کمپیوٹر پر اپنے نگرانی ہینڈل ختم کر لیے ہیں (126، حد 128)، اس لیے فائلوں کی نگرانی کرنے والے پروگرام “Too many open files” کے ساتھ ناکام ہو جاتے ہیں۔ سب سے زیادہ ہینڈل node (پروسیس 4242) کے پاس ہیں: اس پروگرام کو دوبارہ شروع کریں، یا کسی منتظم سے fs.inotify.max_user_instances بڑھانے کو کہیں۔
- `instances_at_limit` / `other`: آپ کے اکاؤنٹ نے اس کمپیوٹر پر اپنے نگرانی ہینڈل ختم کر لیے ہیں (126، حد 128)، اس لیے فائلوں کی نگرانی کرنے والے پروگرام “Too many open files” کے ساتھ ناکام ہو جاتے ہیں۔ کسی منتظم سے fs.inotify.max_user_instances بڑھانے کو کہیں۔
