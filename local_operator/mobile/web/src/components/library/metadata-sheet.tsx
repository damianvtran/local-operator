/**
 * Edit one asset's metadata: name, description and tags.
 *
 * The form follows the projects sheet's rules because they were already settled
 * for this phone: a `busy` latch for the whole write, the daemon's own sentence
 * on refusal, and a RE-READ after the write rather than a local guess. Here that
 * last rule matters twice — the relay's effective name is `user > librarian >
 * derived`, so only another read can tell you what actually won.
 *
 * TAG NAMES ARE GRAMMAR-CHECKED BEFORE THE REQUEST. The store's own grammar is
 * `^[a-z0-9][a-z0-9_-]{0,23}$`; a client that sends a name the grammar rejects
 * paints a create button that always fails, so the same regex gates the request.
 * Existing tags are toggled by id; a new name is created first and then joins
 * the selection, which keeps one write path (the store's `create_tag`).
 */
import { useEffect, useState } from "react";
import {
	createLibraryTag,
	getLibraryAsset,
	getLibraryTags,
	patchLibraryAsset,
	setLibraryTags,
} from "../../api";
import { TAG_NAME_RE } from "../../lib/library";
import type { LibraryAsset, LibraryTag } from "../../types";
import { Button } from "../ui/button";
import { Chip } from "../ui/chip";
import { Sheet } from "../ui/sheet";

const FIELD_CLASS =
	"min-h-11 w-full rounded-sm border border-control bg-surface px-3 text-body text-ink outline-none placeholder:text-ink-dim";

export function MetadataSheet({
	open,
	asset,
	onClose,
	onSaved,
}: {
	open: boolean;
	asset: LibraryAsset;
	onClose: () => void;
	onSaved: (asset: LibraryAsset) => void;
}) {
	const [name, setName] = useState(asset.name);
	const [description, setDescription] = useState(asset.description);
	const [selected, setSelected] = useState<string[]>(asset.tags.map((tag) => tag.id));
	const [available, setAvailable] = useState<LibraryTag[]>(asset.tags);
	const [newTag, setNewTag] = useState("");
	const [busy, setBusy] = useState(false);
	const [error, setError] = useState("");

	/* Re-seeded whenever the sheet OPENS against a fresh row: the asset may have
	   been edited elsewhere since the screen read it, and a stale form that
	   silently saves over that edit is the failure mode this avoids. */
	useEffect(() => {
		if (!open) return;
		setName(asset.name);
		setDescription(asset.description);
		setSelected(asset.tags.map((tag) => tag.id));
		setAvailable(asset.tags);
		setNewTag("");
		setError("");
	}, [open, asset]);

	useEffect(() => {
		if (!open) return;
		let live = true;
		getLibraryTags()
			.then((payload) => {
				if (!live) return;
				/* Union, not replacement: a tag the row carries but the list has
				   not caught up to yet must still be visible and stay selected. */
				setAvailable((current) => {
					const byId = new Map(current.map((tag) => [tag.id, tag]));
					for (const tag of payload.tags) byId.set(tag.id, tag);
					return [...byId.values()].sort((a, b) => a.name.localeCompare(b.name));
				});
			})
			.catch(() => {
				/* The row's own tags remain editable; loading the whole vocabulary
				   is a convenience, never a prerequisite for the save. */
			});
		return () => {
			live = false;
		};
	}, [open]);

	const toggle = (tagId: string) => {
		setSelected((current) =>
			current.includes(tagId) ? current.filter((id) => id !== tagId) : [...current, tagId],
		);
	};

	const addTag = async () => {
		if (busy) return;
		const candidate = newTag.trim();
		if (!TAG_NAME_RE.test(candidate)) {
			setError("Tag names use lowercase letters, digits, underscore and hyphen (max 24).");
			return;
		}
		if (available.some((tag) => tag.name === candidate)) {
			setError("That tag already exists.");
			return;
		}
		setBusy(true);
		setError("");
		try {
			const { tag } = await createLibraryTag(candidate);
			setAvailable((current) => [...current, tag].sort((a, b) => a.name.localeCompare(b.name)));
			setSelected((current) => [...current, tag.id]);
			setNewTag("");
		} catch (e) {
			setError(String((e as Error).message ?? e));
		} finally {
			setBusy(false);
		}
	};

	const save = async () => {
		if (busy) return;
		if (!name.trim()) {
			setError("A name is required.");
			return;
		}
		setBusy(true);
		setError("");
		try {
			await patchLibraryAsset(asset.id, {
				name: name.trim(),
				description: description.trim(),
			});
			await setLibraryTags(asset.id, selected);
			// Re-read, never patch locally: the relay decides the effective value.
			const fresh = await getLibraryAsset(asset.id);
			onSaved(fresh);
			onClose();
		} catch (e) {
			setError(String((e as Error).message ?? e));
		} finally {
			setBusy(false);
		}
	};

	return (
		<Sheet open={open} onClose={onClose} title="edit asset">
			<div className="flex flex-col gap-3 p-3">
				<label className="flex flex-col gap-1">
					<span className="text-meta text-ink-muted">name</span>
					<input
						value={name}
						onChange={(event) => setName(event.target.value)}
						maxLength={300}
						className={FIELD_CLASS}
					/>
				</label>
				<label className="flex flex-col gap-1">
					<span className="text-meta text-ink-muted">description</span>
					<textarea
						value={description}
						onChange={(event) => setDescription(event.target.value)}
						rows={3}
						maxLength={4000}
						className={`${FIELD_CLASS} py-2`}
					/>
				</label>

				<section className="flex flex-col gap-1">
					<h2 className="text-meta text-ink-muted">tags</h2>
					<div className="flex flex-wrap gap-1">
						{available.map((tag) => (
							<Chip
								key={tag.id}
								aria-pressed={selected.includes(tag.id)}
								selected={selected.includes(tag.id)}
								onClick={() => toggle(tag.id)}
							>
								{tag.name}
							</Chip>
						))}
					</div>
					<div className="flex items-end gap-1">
						<input
							value={newTag}
							onChange={(event) => setNewTag(event.target.value)}
							placeholder="new tag name"
							aria-label="new tag name"
							className={FIELD_CLASS}
						/>
						<Button variant="outline" onClick={() => void addTag()} disabled={busy}>
							add
						</Button>
					</div>
				</section>

				{error ? (
					<p role="alert" className="text-body-sm text-danger">
						{error}
					</p>
				) : null}

				<Button variant="primary" onClick={() => void save()} disabled={busy} aria-busy={busy}>
					{busy ? "saving…" : "save"}
				</Button>
			</div>
		</Sheet>
	);
}
