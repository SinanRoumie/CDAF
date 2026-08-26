"""Post-hoc, read-only analysis passes over finished CDAF rounds.

Nothing here changes legality, tabula-rasa behavior, the judge, or any upstream
state -- every pass reruns the UNMODIFIED judge on copies of a completed round.
"""
