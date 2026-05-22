module.exports = {
  apps: [
    {
      name: "stock-dca-backtest",
      script: "./start_streamlit.sh",
      cwd: __dirname,
      interpreter: "bash",
      // Auto-restart on crash
      autorestart: true,
      watch: false,
      max_restarts: 10,
      restart_delay: 3000,
      // Logs
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      error_file: "./logs/err.log",
      out_file: "./logs/out.log",
      merge_logs: true,
      // Graceful shutdown
      kill_timeout: 5000,
    },
  ],
};
