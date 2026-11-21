// The race's fleet pair (lead, 2026-11-21 19:00Z): 1 A100 against 8 from #75's interim fleet group, and Atlas's
// cleared line built from the two rows shown, so the words never drift from the bars. null if the data cannot back it.
export function fleetPair(g) {
  const one = g?.rows?.find((r) => r.id === "fleet-1"), eight = g?.rows?.find((r) => r.id === "fleet-8");
  if (!one || !eight || ![one, eight].every((r) => r.label === "MEASURED" && Number.isFinite(r.seconds) && r.seconds > 0)) return null;
  const card = eight.gpu?.match(/^8x NVIDIA (\S+)$/)?.[1];
  if (!card || one.gpu !== `1x NVIDIA ${card}` || !/all 1000 studies$/.test(one.method) ||
      !/re-imaging\/feature-generation .*outputs bit-identical/.test(g.title)) return null;
  const line = `MEASURED, observed ${(one.seconds / eight.seconds).toFixed(1)}× compute-span reduction ` +
    `${one.seconds.toFixed(1)}→${eight.seconds.toFixed(1)} s for 1,000 Charville re-imaging/feature-generation studies, ` +
    `8 ${card} vs 1, one sweep; outputs bit-identical; inference outside timing`;
  return { rows: [one, eight], line };
}
