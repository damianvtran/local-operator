/**
 * Library list (`#/library`).
 *
 * A 2-column tile grid inside the phone's 28rem column. Three patterns are
 * inherited rather than reinvented:
 *
 * - **The server sorts and filters.** Search is the relay's own `/api/library`,
 *   debounced 180 ms with a sequence guard (the past-sessions screen's pattern);
 *   a stale answer can never overwrite a newer query. Paging is the cursor the
 *   relay mints, and `cursor_reset` means "the catalog moved, restart" — the
 *   service's own rule, not a client guess.
 * - **Filters collapse.** A phone cannot hold the desktop's filter rail, so
 *   kind is a chip row (the one filter reached often) and everything else —
 *   origin, tag, highlighted, unorganized, favorites — lives in one Sheet.
 * - **Nothing is derived locally.** Counts, ordering and the origin group come
 *   from the frame; the screen only groups and paints them.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { getLibrary, libraryBytesUrl, patchLibraryAsset } from "../api";
import { AssetCard } from "../components/library/asset-card";
import { Chip } from "../components/ui/chip";
import { Sheet } from "../components/ui/sheet";
import { cn } from "../lib/cn";
import { navigate } from "../router";
import { messageRuntime } from "../store";
import type { LibraryAsset, LibraryListing } from "../types";

type SortId = "recent" | "name" | "kind" | "size" | "created" | "modified";

const SORTS: { id: SortId; label: string }[] = [
	{ id: "recent", label: "recently seen" },
	{ id: "name", label: "name" },
	{ id: "kind", label: "kind" },
	{ id: "size", label: "size" },
	{ id: "created", label: "created" },
	{ id: "modified", label: "modified" },
];

/** The chip row's order is the classifier's own display order (the design names
    it as the filter-rail order), filtered to kinds the catalog actually holds. */
const CATEGORY_ORDER = ["image", "sound", "document", "presentation", "video", "spreadsheet", "code", "other"];
const CATEGORY_LABEL: Record<string, string> = {
	image: "images",
	sound: "audio",
	document: "docs",
	presentation: "slides",
	video: "video",
	spreadsheet: "sheets",
	code: "code",
	other: "other",
};
const ORIGIN_OPTIONS = [
	{ id: "", label: "all" },
	{ id: "generated", label: "generated" },
	{ id: "created", label: "created" },
	{ id: "mentioned", label: "mentioned" },
];

type Filters = {
	category: string;
	origin: string;
	tag: string;
	highlighted: boolean;
	organized: boolean;
	favorite: boolean;
	sort: SortId;
};

const EMPTY_FILTERS: Filters = {
	category: "",
	origin: "",
	tag: "",
	highlighted: false,
	organized: false,
	favorite: false,
	sort: "recent",
};

function activeFilterCount(filters: Filters): number {
	return [
		filters.origin,
		filters.tag,
		filters.highlighted,
		filters.organized,
		filters.favorite,
	].filter(Boolean).length;
}

export function LibraryScreen() {
	const [query, setQuery] = useState("");
	const [filters, setFilters] = useState<Filters>(EMPTY_FILTERS);
	const [listing, setListing] = useState<LibraryListing | null>(null);
	const [rows, setRows] = useState<LibraryAsset[]>([]);
	const [loading, setLoading] = useState(true);
	const [paging, setPaging] = useState(false);
	const [error, setError] = useState("");
	const [sortOpen, setSortOpen] = useState(false);
	const [filtersOpen, setFiltersOpen] = useState(false);
	const [menuAsset, setMenuAsset] = useState<LibraryAsset | null>(null);
	const [notice, setNotice] = useState("");
	const seqRef = useRef(0);

	const fetchPage = useCallback(
		async (cursor?: string) => {
			const seq = ++seqRef.current;
			if (cursor) setPaging(true);
			else setLoading(true);
			try {
				const frame = await getLibrary({
					q: query.trim() || undefined,
					categories: filters.category ? [filters.category] : undefined,
					origin_groups: filters.origin ? [filters.origin] : undefined,
					tag: filters.tag || undefined,
					highlighted: filters.highlighted || undefined,
					// The relay's filter is "is organized"; the control's word is
					// "unorganized", so the request carries the inverse explicitly.
					organized: filters.organized ? false : undefined,
					favorite: filters.favorite || undefined,
					sort: filters.sort,
					cursor,
				});
				if (seq !== seqRef.current) return;
				setListing(frame);
				setRows((current) =>
					cursor && !frame.cursor_reset ? [...current, ...frame.rows] : frame.rows,
				);
				setError("");
			} catch (e) {
				if (seq !== seqRef.current) return;
				setError(String((e as Error).message ?? e));
			} finally {
				if (seq === seqRef.current) {
					setLoading(false);
					setPaging(false);
				}
			}
		},
		[filters, query],
	);

	useEffect(() => {
		const timer = setTimeout(() => void fetchPage(), 180);
		return () => clearTimeout(timer);
	}, [fetchPage]);

	/* The catalog stops at a budget and answers "building" with honest progress.
	   A short poll is the cheapest correct finish: the index writes on its own
	   side, and the screen must not spin a second builder. */
	useEffect(() => {
		if (listing?.state !== "building") return;
		const timer = setTimeout(() => void fetchPage(), 2500);
		return () => clearTimeout(timer);
	}, [listing?.state, fetchPage]);

	const patchFavorite = async (asset: LibraryAsset) => {
		setMenuAsset(null);
		try {
			await patchLibraryAsset(asset.id, { favorite: !asset.favorite });
			await fetchPage();
			setNotice(asset.favorite ? "Removed from favorites." : "Added to favorites.");
		} catch (e) {
			setNotice(String((e as Error).message ?? e));
		}
	};

	const copyId = async (asset: LibraryAsset) => {
		setMenuAsset(null);
		try {
			await navigator.clipboard.writeText(asset.id);
			setNotice("Asset id copied.");
		} catch {
			setNotice(`Asset id: ${asset.id}`);
		}
	};

	const openSource = (asset: LibraryAsset) => {
		const occurrence = asset.occurrences[0];
		setMenuAsset(null);
		if (occurrence) navigate(`/s/${encodeURIComponent(occurrence.session_id)}`);
	};

	const askAbout = (asset: LibraryAsset) => {
		setMenuAsset(null);
		navigate(`/library/a/${encodeURIComponent(asset.id)}`);
	};

	const categoryChips = [
		{ id: "", label: "all", count: listing?.total ?? 0 },
		...CATEGORY_ORDER.filter((category) => (listing?.counts.category[category] ?? 0) > 0).map(
			(category) => ({
				id: category,
				label: CATEGORY_LABEL[category] ?? category,
				count: listing?.counts.category[category] ?? 0,
			}),
		),
	];

	return (
		<div className="mx-auto flex h-dvh w-full max-w-[var(--lo-column-max,28rem)] flex-col overflow-hidden">
			<header className="flex items-center gap-1 px-2 pt-[max(env(safe-area-inset-top),0.75rem)] pb-1">
				<button
					type="button"
					onClick={() => navigate("/")}
					aria-label="back to sessions"
					className="flex min-h-11 min-w-11 items-center justify-center rounded-sm text-ink-muted active:bg-elevated"
				>
					‹
				</button>
				<h1 className="min-w-0 flex-1 truncate px-1 text-heading text-ink">library</h1>
				<button
					type="button"
					onClick={() => setSortOpen(true)}
					aria-label="sort"
					className="flex min-h-11 items-center justify-center rounded-md border border-control bg-surface px-2 text-meta text-ink-muted active:bg-elevated"
				>
					sort
				</button>
				<button
					type="button"
					onClick={() => setFiltersOpen(true)}
					aria-label="filters"
					className={cn(
						"flex min-h-11 items-center justify-center gap-1 rounded-md border px-2 text-meta active:bg-elevated",
						activeFilterCount(filters)
							? "border-accent bg-accent-wash text-accent"
							: "border-control bg-surface text-ink-muted",
					)}
				>
					filter
					{activeFilterCount(filters) ? (
						<span className="rounded-full bg-accent px-1 text-on-accent">{activeFilterCount(filters)}</span>
					) : null}
				</button>
			</header>

			<div className="px-2 pb-2">
				<input
					type="search"
					value={query}
					onChange={(event) => setQuery(event.target.value)}
					placeholder="search assets…"
					aria-label="search assets"
					className="min-h-11 w-full rounded-md border border-control bg-surface px-3 text-body text-ink outline-none placeholder:text-ink-dim"
				/>
			</div>

			<div className="lo-scroll flex gap-1 overflow-x-auto px-2 pb-2">
				{categoryChips.map((chip) => (
					<Chip
						key={chip.id || "all"}
						aria-pressed={filters.category === chip.id}
						selected={filters.category === chip.id}
						onClick={() => setFilters((current) => ({ ...current, category: chip.id }))}
						className="shrink-0"
					>
						{chip.label}
						<span className="text-ink-dim">{chip.count}</span>
					</Chip>
				))}
			</div>

			{listing?.state === "building" ? (
				<p role="status" className="mx-2 mb-2 rounded-sm border border-warning-border bg-warning-wash px-2 py-1 text-meta text-warning">
					building the index — {listing.scan.sessions_scanned} of {listing.scan.sessions_total} conversations
					scanned
				</p>
			) : null}
			{listing?.cursor_reset ? (
				<p role="status" className="mx-2 mb-2 text-meta text-ink-dim">
					the catalog changed; showing the first page again
				</p>
			) : null}
			{notice ? (
				<p role="status" className="mx-2 mb-2 text-meta text-ink-muted">
					{notice}
				</p>
			) : null}

			<main className="lo-scroll min-h-0 flex-1 overflow-y-auto px-2 pb-4">
				{loading && rows.length === 0 ? (
					<div className="grid grid-cols-2 gap-2" aria-label="loading assets" aria-busy="true">
						{Array.from({ length: 6 }, (_, index) => (
							<div key={index} className="aspect-[4/3] animate-pulse rounded-md bg-sunken" />
						))}
					</div>
				) : error ? (
					<div className="flex flex-col items-center gap-2 py-10">
						<p role="alert" className="text-center text-body-sm text-danger">
							{error}
						</p>
						<button
							type="button"
							onClick={() => void fetchPage()}
							className="min-h-11 rounded-md border border-control bg-surface px-3 text-body-sm text-ink active:bg-elevated"
						>
							Try again
						</button>
					</div>
				) : rows.length === 0 ? (
					<div className="flex flex-col items-center gap-1 py-10 text-center">
						<p className="text-body text-ink-muted">
							{query || activeFilterCount(filters) ? "no matching assets" : "no assets yet"}
						</p>
						<p className="text-body-sm text-ink-dim">
							{query || activeFilterCount(filters)
								? "Try a different search or clear the filters."
								: "Files a conversation creates or mentions will appear here."}
						</p>
					</div>
				) : (
					<div className="grid grid-cols-2 gap-2">
						{rows.map((asset) => (
							<AssetCard
								key={asset.id}
								asset={asset}
								onOpen={() => navigate(`/library/a/${encodeURIComponent(asset.id)}`)}
								onMenu={() => setMenuAsset(asset)}
							/>
						))}
					</div>
				)}
				{listing?.next_cursor ? (
					<button
						type="button"
						disabled={paging}
						aria-busy={paging}
						onClick={() => void fetchPage(listing.next_cursor ?? undefined)}
						className="mt-3 flex min-h-11 w-full items-center justify-center rounded-md border border-control bg-surface text-body-sm text-ink-muted active:bg-elevated disabled:text-ink-disabled"
					>
						{paging ? "loading…" : "load more"}
					</button>
				) : null}
				{!loading && rows.length > 0 ? (
					<p className="pt-2 text-center text-meta text-ink-dim">
						{messageRuntime.t("relay.library.count", {
							count: listing?.total ?? rows.length,
						})}
					</p>
				) : null}
			</main>

			<Sheet open={sortOpen} onClose={() => setSortOpen(false)} title="sort">
				<div className="flex flex-col p-2">
					{SORTS.map((option) => (
						<button
							key={option.id}
							type="button"
							onClick={() => {
								setFilters((current) => ({ ...current, sort: option.id }));
								setSortOpen(false);
							}}
							aria-pressed={filters.sort === option.id}
							className="flex min-h-11 items-center justify-between gap-2 rounded-sm px-2 text-left text-body text-ink active:bg-surface"
						>
							{option.label}
							{filters.sort === option.id ? (
								<span className="text-accent" aria-hidden>
									✓
								</span>
							) : null}
						</button>
					))}
				</div>
			</Sheet>

			<Sheet open={filtersOpen} onClose={() => setFiltersOpen(false)} title="filters">
				<div className="flex flex-col gap-4 p-2">
					<section className="flex flex-col gap-1">
						<h2 className="text-meta text-ink-muted">origin</h2>
						<div className="flex flex-wrap gap-1">
							{ORIGIN_OPTIONS.map((option) => (
								<Chip
									key={option.id || "all"}
									aria-pressed={filters.origin === option.id}
									selected={filters.origin === option.id}
									onClick={() => setFilters((current) => ({ ...current, origin: option.id }))}
								>
									{option.label}
								</Chip>
							))}
						</div>
					</section>

					<section className="flex flex-col gap-1">
						<h2 className="text-meta text-ink-muted">state</h2>
						<div className="flex flex-wrap gap-1">
							<Chip
								aria-pressed={filters.highlighted}
								selected={filters.highlighted}
								onClick={() => setFilters((current) => ({ ...current, highlighted: !current.highlighted }))}
							>
								★ highlighted
							</Chip>
							<Chip
								aria-pressed={filters.organized}
								selected={filters.organized}
								onClick={() => setFilters((current) => ({ ...current, organized: !current.organized }))}
							>
								unorganized
							</Chip>
							<Chip
								aria-pressed={filters.favorite}
								selected={filters.favorite}
								onClick={() => setFilters((current) => ({ ...current, favorite: !current.favorite }))}
							>
								favorites
							</Chip>
						</div>
					</section>

					<section className="flex flex-col gap-1">
						<h2 className="text-meta text-ink-muted">tags</h2>
						{listing?.tags.length ? (
							<div className="flex flex-wrap gap-1">
								<Chip
									aria-pressed={!filters.tag}
									selected={!filters.tag}
									onClick={() => setFilters((current) => ({ ...current, tag: "" }))}
								>
									any
								</Chip>
								{listing.tags.map((tag) => (
									<Chip
										key={tag.id}
										aria-pressed={filters.tag === tag.id}
										selected={filters.tag === tag.id}
										onClick={() =>
											setFilters((current) => ({
												...current,
												tag: current.tag === tag.id ? "" : tag.id,
											}))
										}
									>
										{tag.name}
										<span className="text-ink-dim">{tag.count ?? 0}</span>
									</Chip>
								))}
							</div>
						) : (
							<p className="text-meta text-ink-dim">No tags yet. Add one from an asset&apos;s details.</p>
						)}
					</section>

					<button
						type="button"
						onClick={() =>
							/* Clearing narrows the CATALOG, not the reader's chosen view.
							   Sort is a display preference and survives; resetting it to the
							   default made a filter action silently reorder the grid
							   (UX round 1, U2). */
							setFilters((current) => ({ ...EMPTY_FILTERS, sort: current.sort }))
						}
						className="min-h-11 rounded-md border border-control bg-surface px-3 text-body-sm text-ink-muted active:bg-elevated"
					>
						clear filters
					</button>
				</div>
			</Sheet>

			<Sheet
				open={menuAsset !== null}
				onClose={() => setMenuAsset(null)}
				title={menuAsset?.name || "asset"}
			>
				{menuAsset ? (
					<div className="flex flex-col p-2">
						<button
							type="button"
							onClick={() => askAbout(menuAsset)}
							className="flex min-h-11 items-center rounded-sm px-2 text-left text-body text-ink active:bg-surface"
						>
							ask about this asset
						</button>
						{menuAsset.occurrences[0] ? (
							<button
								type="button"
								onClick={() => openSource(menuAsset)}
								className="flex min-h-11 items-center rounded-sm px-2 text-left text-body text-ink active:bg-surface"
							>
								open source conversation
							</button>
						) : null}
						<button
							type="button"
							onClick={() => void patchFavorite(menuAsset)}
							className="flex min-h-11 items-center rounded-sm px-2 text-left text-body text-ink active:bg-surface"
						>
							{menuAsset.favorite ? "remove from favorites" : "add to favorites"}
						</button>
						<a
							href={libraryBytesUrl(menuAsset.id, true)}
							onClick={() => setMenuAsset(null)}
							className="flex min-h-11 items-center rounded-sm px-2 text-left text-body text-ink active:bg-surface"
						>
							download
						</a>
						<button
							type="button"
							onClick={() => void copyId(menuAsset)}
							className="flex min-h-11 items-center rounded-sm px-2 text-left text-body text-ink active:bg-surface"
						>
							copy asset id
						</button>
					</div>
				) : null}
			</Sheet>
		</div>
	);
}
