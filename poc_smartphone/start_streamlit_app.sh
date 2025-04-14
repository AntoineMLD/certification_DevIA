#!/bin/bash
echo "===== Démarrage de l'application Streamlit de reconnaissance de gravures ====="

# Vérifier si Python est installé
if ! command -v python3 &> /dev/null; then
    echo "Python n'est pas installé. Veuillez installer Python 3.8 ou supérieur."
    exit 1
fi

# Vérifier si pip est installé
if ! command -v pip3 &> /dev/null; then
    echo "pip n'est pas installé. Veuillez installer pip."
    exit 1
fi

# Installer les dépendances si nécessaire
echo "Installation des dépendances..."
pip3 install -r requirements_streamlit.txt

# Afficher l'adresse IP pour se connecter depuis un smartphone
echo ""
echo "Pour accéder à l'application depuis un smartphone:"
echo "1. Assurez-vous que le PC et le smartphone sont sur le même réseau WiFi"
echo "2. Utilisez l'une des adresses suivantes sur votre smartphone:"

# Afficher les adresses IP selon le système d'exploitation
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    ifconfig | grep "inet " | grep -v 127.0.0.1
else
    # Linux
    ip addr show | grep "inet " | grep -v 127.0.0.1
fi

echo ""
echo "Puis ajoutez :8501 à l'adresse IP (exemple: 192.168.1.100:8501)"
echo ""

# Lancer l'application Streamlit
echo "Démarrage de l'application Streamlit..."
echo ""
streamlit run streamlit_draw_app.py 