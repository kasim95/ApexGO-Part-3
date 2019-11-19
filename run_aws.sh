export PATH=${PATH}:/usr/home/ApexGO-Part-3/

BOT_USERNAME=ApexGO_BOT
BOT_APIKEY=
BOARDSIZE=19

if [ ${BOT_APIKEY}"x" = "x" ]; then
	echo "Error: must provide Bot's API key"
	echo "This is not provided inside git repository"
	
	exit 1
fi

echo "Initializing bot ..."


forever start gtp2ogs.js --username ${BOT_USERNAME} --apikey ${BOT_APIKEY} --hidden --persist --boardsize ${BOARDSIZE} --debug -- python3 run_gtp_ac.py

if [ ${?} -eq 0 ]; then
	echo "failed to launch bot!" >&2
	exit ${?}
fi

echo "Bot session active"


