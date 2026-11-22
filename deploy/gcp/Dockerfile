# The static demo/ behind nginx on Cloud Run. scripts/deploy-gcp.sh builds the context: this file, nginx.conf, site/.
FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY site/ /usr/share/nginx/html/
EXPOSE 8080
