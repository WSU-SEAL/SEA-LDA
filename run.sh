


bash clean.sh


# Run GraphQL Posts
python -Wignore SEA_LDA.py --maxtopic 16 --mintopic 5 --steptopic 1 --repetitions 6 --miniteration 1000 --maxiteration 1001 --stepiteration 5 --modelname "graphql" --sourcefileformat "xlsx"


# Run Refactor Posts
# python SEA_LDA.py --maxtopic 50 --mintopic 5 --steptopic 5 --repetitions 10 --miniteration 500 --maxiteration 2000 --stepiteration 500 --modelname "refactor"


# Run IoT Posts
#python SEA_LDA.py --maxtopic 6 --mintopic 5 --steptopic 5 --repetitions 1 --miniteration 10 --maxiteration 11 --stepiteration 10 --modelname "iot" --sourcefileformat "xml"