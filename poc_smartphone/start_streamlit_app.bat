@echo off
echo ===== Demarrage de l'application Streamlit de reconnaissance de gravures =====

REM Verifier si Python est installe
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo Python n'est pas installe. Veuillez installer Python 3.8 ou superieur.
    exit /b 1
)

REM Verifier si pip est installe
pip --version > nul 2>&1
if %errorlevel% neq 0 (
    echo pip n'est pas installe. Veuillez installer pip.
    exit /b 1
)

REM Installer les dependances si necessaire
echo Installation des dependances...
pip install -r requirements_streamlit.txt

REM Afficher l'adresse IP pour se connecter depuis un smartphone
echo.
echo Pour acceder a l'application depuis un smartphone:
echo 1. Assurez-vous que le PC et le smartphone sont sur le meme reseau WiFi
echo 2. Utilisez l'une des adresses suivantes sur votre smartphone:
ipconfig | findstr /i "IPv4"
echo.

REM Lancer l'application Streamlit
echo Demarrage de l'application Streamlit...
echo.
streamlit run streamlit_draw_app.py 