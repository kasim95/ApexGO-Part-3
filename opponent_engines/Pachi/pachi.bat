rem Batch file to run Pachi.
rem For convenience mostly, in case you use lots of options.

rem If your Go Program doesn't let you run .bat files directly give it cmd.exe:
rem   C:\Windows\System32\cmd.exe /c C:\path\to\pachi.bat

@echo off
cd %~dp0


rem ************************************************************************
rem Time Settings:

rem 20s per move.
rem pachi.exe -t 20

rem 10 minutes sudden death.
rem pachi.exe -t _600

rem 5000 playouts per move.
rem pachi.exe -t =5000

rem Think more when needed: max 15k playouts per move.
rem pachi.exe -t =5000:15000

rem Don't think too much during fuseki.
rem pachi.exe -t =5000:15000 --fuseki-time =4000


rem ************************************************************************
rem Fixed Strength:

rem kgs 2d:
rem pachi.exe -t =5000:15000

rem kgs 3k:
rem pachi.exe -t =5000 --nodcnn


rem ************************************************************************
rem Other Options:

rem resign when winrate < 25%.
rem pachi.exe resign_threshold=0.25

rem Use one cpu thread only.
rem pachi.exe threads=1

rem Save log to file.
rem pachi.exe 2>pachi_log.txt

rem Play without dcnn with time settings 20:00 S.D. on 8 threads, taking
rem up to 3Gb of memory, and thinking during the opponent's turn as well.
rem pachi.exe -t _1200 --nodcnn threads=8,max_tree_size=3072,pondering


rem ************************************************************************


pachi.exe -t =5000:15000 --fuseki-time =4000 max_tree_size=100  
