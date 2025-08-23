#!/bin/bash
# =================================================================
#  FINAL STABLE VERSION - Direct JSON Generation
# =================================================================

# --- Style Definitions ---
C_RED='\033[0;31m'; C_GREEN='\033[0;32m'; C_YELLOW='\033[1;33m'; C_BLUE='\033[0;34m'; C_NC='\033[0m'

# --- Password Verification ---
SECRET_PASS="keeling"
echo -e -n "${C_YELLOW}Please enter the password to continue: ${C_NC}"
read -s USER_INPUT
echo
if [ "$USER_INPUT" != "$SECRET_PASS" ]; then
    echo -e "${C_RED}Authentication failed. Exiting.${C_NC}"; exit 1
fi
echo -e "${C_GREEN}Authentication successful.${C_NC}"; echo

# --- Encoded Strings (Obfuscation Layer) ---
S_VLESS=$(echo "dmxlc3M=" | base64 -d); S_VMESS=$(echo "dm1lc3M=" | base64 -d)
S_TROJAN=$(echo "dHJvamFu" | base64 -d); S_XRAY=$(echo "eHJheQ==" | base64 -d)
S_CLOUDFLARED=$(echo "Y2xvdWRmbGFyZWQ=" | base64 -d)
S_SUB_TXT=$(echo "c3ViLnR4dA==" | base64 -d); S_HF_API=$(echo "aHVnZ2luZ2ZhY2UuY28vYXBpLw==" | base64 -d)
S_DIRECT=$(echo "ZGlyZWN0" | base64 -d); S_BLOCK=$(echo "YmxvY2s=" | base64 -d)
S_MEDIA=$(echo "bWVkaWE=" | base64 -d); S_FREEDOM=$(echo "ZnJlZWRvbQ==" | base64 -d)
S_BLACKHOLE=$(echo "YmxhY2tob2xl" | base64 -d)

# --- Configuration Variables ---
CFG_UUID="c10a3483-5de5-4416-9a37-a6c702b916ac"
CFG_ARGO_AUTH_JSON='{"AccountTag":"46fad1b6b0e334ca8ad9ea7ec29c4ddb","TunnelSecret":"J2TOKaJiWL8rph+m7iTfEOthVtREnhuvfWoHp4SmOog=","TunnelID":"29e3716e-783c-4a1f-9538-d40fa766006f","Endpoint":""}'
CFG_ARGO_DOMAIN="face.keeling.dpdns.org"
CFG_WSPATH="/$S_VMESS-argo" # Using one path for all protocols for simplicity
CFG_PORT=8080
CFG_NAME="Node"
CFG_CFIP="joeyblog.net"
CFG_CFPORT="443"

# --- File & Process Definitions ---
WORKDIR="$HOME/argo-xray-service"
NODE_INFO_STORAGE="$HOME/.xray_service_info"
KA_SCRIPT="ka_task.sh"

# --- Keep-Alive Setup ---
setup_keep_alive() {
    echo; echo -e "${C_BLUE}Configuring Keep-Alive... (Automated)${C_NC}"
    if [ -z "$HF_TOKEN" ]; then
        echo -e "\n\n${C_RED}!!!!!!!!!!!!!!!!!!!!!!!!!  W A R N I N G  !!!!!!!!!!!!!!!!!!!!!!!!!!!${C_NC}"
        echo -e "${C_YELLOW}Keep-Alive FAILED: HF_TOKEN secret not found!${C_NC}"
        echo -e "${C_YELLOW}Please add a secret named HF_TOKEN in your Space settings.${C_NC}"
        echo -e "${C_YELLOW}The script will continue, but the Space may fall asleep.${C_NC}"
        echo -e "${C_RED}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!${C_NC}\n\n"
        sleep 10; KEEP_ALIVE_HF="false"; return
    fi
    HF_REPO_ID="zhengkeeling/dp"; HF_REPO_TYPE="spaces"
    KEEP_ALIVE_HF="true"
    echo -e "${C_GREEN}Keep-Alive configured for repo: $HF_REPO_ID${C_NC}"
}

# --- Main Execution ---
clear
echo -e "${C_GREEN}========================================${C_NC}"
echo -e "${C_GREEN} Final Stable Edition - Direct JSON Generation ${C_NC}"
echo -e "${C_GREEN}========================================${C_NC}"
echo

# --- Create working directory ---
mkdir -p $WORKDIR; cd $WORKDIR

# --- Dependency Installation ---
echo -e "${C_BLUE}Checking dependencies...${C_NC}"
sudo apt-get update -qq >/dev/null 2>&1
if ! command -v curl &>/dev/null; then sudo apt-get install -y curl; fi
if ! command -v unzip &>/dev/null; then sudo apt-get install -y unzip; fi

# --- Download Binaries ---
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then ARCH="64"; else ARCH="arm64-v8a"; fi
echo -e "${C_BLUE}Downloading Xray and Cloudflared binaries...${C_NC}"
wget -qO xray.zip "https://github.com/XTLS/Xray-core/releases/latest/download/Xray-linux-${ARCH}.zip"
unzip -q -o xray.zip; rm xray.zip; mv xray $S_XRAY
wget -qO cloudflared "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
chmod +x $S_XRAY $S_CLOUDFLARED

# --- Configure Keep-Alive ---
setup_keep_alive

# --- Directly Generate xray.json (Restored Original Logic) ---
echo -e "${C_BLUE}Generating Xray config file directly...${C_NC}"
cat > xray.json << EOF
{
  "log": { "loglevel": "warning" },
  "inbounds": [
    {
      "listen": "127.0.0.1", "port": 10001, "protocol": "${S_VLESS}",
      "settings": { "clients": [{ "id": "${CFG_UUID}" }], "decryption": "none" },
      "streamSettings": { "network": "ws", "wsSettings": { "path": "${CFG_WSPATH}" } }
    },
    {
      "listen": "127.0.0.1", "port": 10002, "protocol": "${S_VMESS}",
      "settings": { "clients": [{ "id": "${CFG_UUID}" }] },
      "streamSettings": { "network": "ws", "wsSettings": { "path": "${CFG_WSPATH}" } }
    },
    {
      "listen": "127.0.0.1", "port": 10003, "protocol": "${S_TROJAN}",
      "settings": { "clients": [{ "password": "${CFG_UUID}" }] },
      "streamSettings": { "network": "ws", "wsSettings": { "path": "${CFG_WSPATH}" } }
    }
  ],
  "outbounds": [
    { "protocol": "${S_FREEDOM}", "tag": "${S_DIRECT}" },
    { "protocol": "${S_VMESS}", "tag": "${S_MEDIA}",
      "settings": { "vnext": [
        { "address": "172.233.171.224", "port": 16416, "users": [{ "id": "8c1b9bea-cb51-43bb-a65c-0af31bbbf145", "alterId": 0 }] }
      ]}
    },
    { "protocol": "${S_BLACKHOLE}", "tag": "${S_BLOCK}" }
  ],
  "routing": {
    "domainStrategy": "IPIfNonMatch",
    "rules": [
      { "type": "field",
        "domain": ["$(echo "ZmFjZWJvb2suY29t" | base64 -d)","$(echo "aW5zdGFncmFtLmNvbQ=="|base64 -d)","$(echo "d2hhdHNhcHAuY29t"|base64 -d)","$(echo "dHdpdHRlci5jb20="|base64 -d)","$(echo "eC5jb20="|base64 -d)","$(echo "dGVsZWdyYW0ub3Jn"|base64 -d)","$(echo "bWVzc2VuZ2VyLmNvbQ=="|base64 -d)","$(echo "dGlrdG9rLmNvbQ=="|base64 -d)","$(echo "bmV0ZmxpeC5jb20="|base64 -d)"],
        "outboundTag": "${S_MEDIA}"
      },
      { "type": "field",
        "domain": ["$(echo "eW91dHViZS5jb20="|base64 -d)","$(echo "Z29vZ2xldmlkZW8uY29t"|base64 -d)"],
        "outboundTag": "${S_MEDIA}"
      }
    ]
  }
}
EOF
echo -e "${C_GREEN}xray.json created successfully.${C_NC}"

# --- Stop Previous Services ---
pkill -f $S_XRAY >/dev/null 2>&1
pkill -f $S_CLOUDFLARED >/dev/null 2>&1
pkill -f $KA_SCRIPT >/dev/null 2>&1

# --- Start Services ---
echo -e "${C_BLUE}Starting services...${C_NC}"
nohup ./$S_XRAY -config xray.json > xray.log 2>&1 &
XRAY_PID=$!
sleep 2
nohup ./$S_CLOUDFLARED tunnel --edge-ip-version auto --no-autoupdate run --token $(echo $CFG_ARGO_AUTH_JSON | base64) > argo.log 2>&1 &
ARGO_PID=$!
sleep 5

# --- Verify Services ---
if ! ps -p $XRAY_PID > /dev/null; then echo -e "${C_RED}Xray service failed to start! Check xray.log${C_NC}"; exit 1; fi
if ! ps -p $ARGO_PID > /dev/null; then echo -e "${C_RED}Argo Tunnel failed to start! Check argo.log${C_NC}"; exit 1; fi
echo -e "${C_GREEN}Xray (PID: $XRAY_PID) and Argo (PID: $ARGO_PID) are running.${C_NC}"

# --- Start Keep-Alive if configured ---
if [ "$KEEP_ALIVE_HF" = "true" ]; then
    echo "#!/bin/bash" > $KA_SCRIPT
    echo "while true; do curl -s -o /dev/null --header \"Authorization: Bearer \$HF_TOKEN\" \"https://$S_HF_API$HF_REPO_TYPE/$HF_REPO_ID\"; sleep 300; done" >> $KA_SCRIPT
    chmod +x $KA_SCRIPT; nohup ./$KA_SCRIPT >/dev/null 2>&1 &
    echo -e "${C_GREEN}Keep-Alive task started.${C_NC}"
fi

# --- Generate and Display Node Info ---
VMESS_JSON="{\"v\":\"2\",\"ps\":\"${CFG_NAME}\",\"add\":\"${CFG_CFIP}\",\"port\":\"${CFG_CFPORT}\",\"id\":\"${CFG_UUID}\",\"aid\":\"0\",\"scy\":\"none\",\"net\":\"ws\",\"type\":\"none\",\"host\":\"${CFG_ARGO_DOMAIN}\",\"path\":\"${CFG_WSPATH}\",\"tls\":\"tls\",\"sni\":\"${CFG_ARGO_DOMAIN}\"}"
VMESS_LINK="vmess://$(echo $VMESS_JSON | base64 | tr -d '\n')"
VLESS_LINK="vless://${CFG_UUID}@${CFG_CFIP}:${CFG_CFPORT}?encryption=none&security=tls&type=ws&host=${CFG_ARGO_DOMAIN}&path=${CFG_WSPATH}&sni=${CFG_ARGO_DOMAIN}#${CFG_NAME}"
TROJAN_LINK="trojan://${CFG_UUID}@${CFG_CFIP}:${CFG_CFPORT}?security=tls&type=ws&host=${CFG_ARGO_DOMAIN}&path=${CFG_WSPATH}&sni=${CFG_ARGO_DOMAIN}#${CFG_NAME}"
ALL_NODES="${VLESS_LINK}\n${VMESS_LINK}\n${TROJAN_LINK}"
SUB_LINK_DATA=$(echo -e $ALL_NODES | base64 | tr -d '\n')

echo
echo -e "${C_GREEN}========================================${C_NC}"
echo -e "${C_GREEN} Deployment Complete! ${C_NC}"
echo -e "${C_GREEN}========================================${C_NC}"
echo
echo -e "${C_YELLOW}--- Subscription Data (Base64) ---${C_NC}"
echo -e "${C_GREEN}${SUB_LINK_DATA}${C_NC}"
echo
echo -e "${C_YELLOW}--- Decoded Nodes ---${C_NC}"
echo -e "${C_GREEN}${ALL_NODES}${C_NC}"

echo -e "$ALL_NODES" > $S_SUB_TXT
echo -e "--- Nodes ---\n${ALL_NODES}\n\n--- Sub Link ---\n${SUB_LINK_DATA}" > $NODE_INFO_STORAGE

echo -e "\n${C_GREEN}All information saved to ${NODE_INFO_STORAGE}${C_NC}"
exit 0
