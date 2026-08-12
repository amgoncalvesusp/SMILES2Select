# Selection zones

Zones are named sets of candidate record IDs. They may come from safe boolean
expressions, descriptor ranges, reference similarity, scaffold properties,
clusters or a visual lasso.

Zones may overlap. Membership is retained in long form, while allocation is
deterministic: explicit priority, then restrictive population, then stable ID
tie-breaking. A molecule is counted once. Unfulfilled quotas are reported and
are never silently relaxed.

Final and reserve allocations are separate. Reserve targets follow the final
zone proportions when sufficient candidates exist; remaining shortfalls are
visible to the user.
