python -m cProfile -o profile.dat alphagozero/alphagozero_rl.py && echo "Generating report ..." && gprof2dot -f pstats profile.dat | dot -Tsvg -o profile.svg