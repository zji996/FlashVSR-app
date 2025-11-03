# Build stage
FROM node:20-alpine AS builder
WORKDIR /app

# 安装 pnpm
RUN corepack enable

# 复制依赖文件并安装
COPY frontend/pnpm-lock.yaml frontend/package.json ./
RUN pnpm install --frozen-lockfile

# 复制源代码并构建
COPY frontend ./
RUN pnpm build

# Runtime stage
FROM nginx:1.27-alpine

# 复制自定义 Nginx 配置
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

# 复制构建产物
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
