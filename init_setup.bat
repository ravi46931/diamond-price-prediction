@echo off
call conda activate ./env
call echo Activating environment env
call pip install -r requirements_dev.txt
call echo Installing the requirements


@REM call python --version
@REM call conda deactivate