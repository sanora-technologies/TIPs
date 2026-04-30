root email: ammarmurtaza010@gmail.com
account name: Shehryar Ahmad
root pwd: sherryHx7@u9vS
instance name : product1
pem key name : dianexea_app.pem

## Steps
`pem ket conversion to private read limited`
chmod 400 dianexea_app.pem
ssh -i dianexea_app.pem ubuntu@13.63.234.199

# Node installation

# Download and install nvm:
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.4/install.sh | bash
# in lieu of restarting the shell
\. "$HOME/.nvm/nvm.sh"
# Download and install Node.js:
nvm install 24
# Verify the Node.js version:
node -v # Should print "v24.14.1".
# Verify npm version:
npm -v # Should print "11.11.0".

sudo apt update -y


# Nginx + MySQL + Git
sudo apt install nginx mysql-server git -y

sudo apt install redis
sudo systemctl start redis
sudo systemctl start mysql

git ssh setup:
ssh-keygen -t ed25519 -C "ec2-server"
cat ~/.ssh/id_ed25519.pub
STEP 3 — Add key to GitHub
Go to GitHub → Settings
Go to SSH and GPG keys
Click New SSH Key
Paste your key
Save
# PM2
npm install -g pm2
<!--  -->
mysql commands:
CREATE USER 'dianexea'@'localhost' IDENTIFIED BY '9a9@f5Gr';
GRANT ALL PRIVILEGES ON *.* TO 'dianexea'@'localhost';
FLUSH PRIVILEGES;
db name: dianexeadb
user: dianexea
db password: 9a9@f5Gr

ssh key name: sanora-github-key

directory maker and cloning:
mkdir -p /var/www
cd /var/www

permission for /var/www: sudo chown -R ubuntu:ubuntu /var/www
1- git clone git@github.com:sanora-technologies/faawebfrontend.git frontend
2- git clone git@github.com:sanora-technologies/DentAIBackend.git backend


backend setup:
update .env
npm install
npm mikro-orm migration:create
npm mikro-orm migration:up
npm run seed:acl
npm run seed:packages
npm run seed:notifications
npm run build



backend:
npm run build


pm2 start /var/www/backend/dist/src/main.js --name dentai-backend -- --port 3001

pm2 save
pm2 startup

frontend:
make build on local npm run build:
rsync -avz --progress \
  -e "ssh -i ~/Downloads/dianexea_app.pem" \
  --exclude 'node_modules' \
  --exclude '.git' \
  --exclude '.env*' \
  .next \
  public \
  package.json \
  package-lock.json \
  next.config.ts \
  ubuntu@13.63.234.199:/var/www/frontend/

pm2 start npm --name faaweb-frontend -- start -- -p 3000
pm2 save
pm2 startup


Nginix Setup:
step1:
# FRONTEND
server {
    listen 80;
    server_name dianexea.com www.dianexea.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;

        proxy_set_header Host $host;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}

# BACKEND (API)
server {
    listen 80;
    server_name api.dianexea.com;

    location / {
        proxy_pass http://localhost:3001;

        proxy_http_version 1.1;

        proxy_set_header Host $host;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}

 sudo ln -s /etc/nginx/sites-available/dentai /etc/nginx/sites-enabled/
 sudo rm /etc/nginx/sites-enabled/default
 sudo nginx -t

 sudo systemctl restart nginx



sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d dianexea.com -d www.dianexea.com -d api.dianexea.com




cd /var/www/faawebfrontend
pm2 restart faaweb-frontend 

pm2 stop all
pm2 restart all


sudo nginx -t
sudo nano /etc/nginx/sites-available/dentai
sudo systemctl reload nginx

    pm2 logs dentai-backend
curl -X POST https://api.dianexea.com/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"alphax1@yopmail.com","password":"11221122"}'
