import brandIcon from "../../assets/brand-icon.png?inline";

/** Exact official white-circle/dark-motif artwork; never recolour it with the
 * theme. Inline it so an identity-proxied tunnel cannot turn an image request
 * into a login redirect. The adjacent heading owns the accessible name; this
 * repeated mark is decorative, not another focus target or extra chrome row.
 * Provenance and the source-byte hash live in docs/design/mobile-relay-guidelines.md. */
export function BrandMark() {
	return <img data-brand-mark src={brandIcon} alt="" aria-hidden="true"
		width="24" height="24" className="size-6 shrink-0" draggable={false} />;
}
