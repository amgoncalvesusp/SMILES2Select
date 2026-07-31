"""Selection Intelligence: turning filter results into a traceable human decision.

This layer never re-runs chemistry. It reads what the pipeline already computed
and helps a person answer one question: among the molecules that passed, which
ones go into the final set?

Its central rule: a chemical result and a human decision are different things
and are stored separately. Selecting or excluding a molecule by hand never
overwrites what the rules said about it.
"""
