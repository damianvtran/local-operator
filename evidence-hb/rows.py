"""Read-only dump of display_bodies rows (message_id, source, conversation) for side-effect proof."""
import sqlite3, sys
p = sys.argv[1]
try:
    c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    cols = [r[1] for r in c.execute("PRAGMA table_info(display_bodies)")]
    rows = c.execute("SELECT * FROM display_bodies").fetchall()
except Exception as e:
    print("display rows: n/a", type(e).__name__, e); sys.exit()
mi, si = cols.index("message_id"), cols.index("source")
print(f"display rows={len(rows)}", sorted((r[mi], r[si]) for r in rows if not str(r[mi]).startswith("bcmsg_coal")),
      "coal:", sum(1 for r in rows if str(r[mi]).startswith("bcmsg_coal")))
