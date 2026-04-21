FROM nginx:alpine

COPY apps/ui/ /usr/share/nginx/html/
COPY infra/nginx/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 3000
