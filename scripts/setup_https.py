#!/usr/bin/env python3
"""
HTTPS Setup Script for Farmer Assistants

This script helps set up HTTPS certificates for the FastAPI assistants.
Supports both self-signed certificates for development and Let's Encrypt for production.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def create_self_signed_cert(cert_dir: Path, domain: str = "localhost"):
    """Create self-signed SSL certificate for development."""
    cert_dir.mkdir(exist_ok=True)
    
    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"
    
    if cert_file.exists() and key_file.exists():
        print(f"SSL certificates already exist in {cert_dir}")
        return cert_file, key_file
    
    print(f"Creating self-signed certificate for {domain}...")
    
    cmd = [
        "openssl", "req", "-x509", "-newkey", "rsa:4096",
        "-keyout", str(key_file),
        "-out", str(cert_file),
        "-days", "365",
        "-nodes",
        "-subj", f"/C=US/ST=CA/L=LAX/O=Farmer/OU=Dev/CN={domain}"
    ]
    
    try:
        subprocess.run(cmd, check=True, cwd=cert_dir.parent)
        print(f"✓ Certificate created: {cert_file}")
        print(f"✓ Private key created: {key_file}")
        return cert_file, key_file
    except subprocess.CalledProcessError as e:
        print(f"Error creating certificate: {e}")
        sys.exit(1)

def setup_lets_encrypt(domain: str, email: str, cert_dir: Path):
    """Set up Let's Encrypt certificate (requires certbot)."""
    print(f"Setting up Let's Encrypt certificate for {domain}...")
    
    # Check if certbot is installed
    try:
        subprocess.run(["certbot", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: certbot is not installed. Install it with:")
        print("  brew install certbot  # on macOS")
        print("  sudo apt install certbot  # on Ubuntu")
        sys.exit(1)
    
    cmd = [
        "certbot", "certonly",
        "--standalone",
        "--email", email,
        "--agree-tos",
        "--no-eff-email",
        "-d", domain
    ]
    
    try:
        subprocess.run(cmd, check=True)
        
        # Copy certificates to our cert directory
        le_cert_dir = Path(f"/etc/letsencrypt/live/{domain}")
        cert_dir.mkdir(exist_ok=True)
        
        subprocess.run(["cp", str(le_cert_dir / "fullchain.pem"), str(cert_dir / "cert.pem")], check=True)
        subprocess.run(["cp", str(le_cert_dir / "privkey.pem"), str(cert_dir / "key.pem")], check=True)
        
        print(f"✓ Let's Encrypt certificate installed in {cert_dir}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error setting up Let's Encrypt: {e}")
        sys.exit(1)

def create_nginx_config(domain: str, hr_port: int, it_port: int, cert_dir: Path):
    """Create nginx reverse proxy configuration for HTTPS termination."""
    
    config = f"""
# Nginx configuration for Farmer Assistants HTTPS
server {{
    listen 443 ssl http2;
    server_name {domain};
    
    ssl_certificate {cert_dir}/cert.pem;
    ssl_certificate_key {cert_dir}/key.pem;
    
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-SHA256:ECDHE-RSA-AES256-SHA384;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # HR Assistant
    location /hr/ {{
        proxy_pass http://127.0.0.1:{hr_port}/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}
    
    # IT Assistant
    location /it/ {{
        proxy_pass http://127.0.0.1:{it_port}/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}
    
    # Default redirect to HR
    location / {{
        return 301 /hr/docs;
    }}
}}

# Redirect HTTP to HTTPS
server {{
    listen 80;
    server_name {domain};
    return 301 https://$server_name$request_uri;
}}
"""
    
    config_file = Path("nginx_farmer_https.conf")
    config_file.write_text(config)
    print(f"✓ Nginx configuration created: {config_file}")
    print("To use this configuration:")
    print(f"  sudo cp {config_file} /etc/nginx/sites-available/farmer")
    print("  sudo ln -s /etc/nginx/sites-available/farmer /etc/nginx/sites-enabled/")
    print("  sudo nginx -t && sudo systemctl reload nginx")

def main():
    parser = argparse.ArgumentParser(description="Set up HTTPS for Farmer Assistants")
    parser.add_argument("--mode", choices=["dev", "prod"], default="dev",
                       help="Development (self-signed) or production (Let's Encrypt)")
    parser.add_argument("--domain", default="localhost",
                       help="Domain name for the certificate")
    parser.add_argument("--email", help="Email for Let's Encrypt (required for prod mode)")
    parser.add_argument("--cert-dir", default="certs", help="Directory to store certificates")
    parser.add_argument("--hr-port", type=int, default=8001, help="HR Assistant port")
    parser.add_argument("--it-port", type=int, default=8002, help="IT Assistant port")
    parser.add_argument("--nginx", action="store_true", help="Generate nginx configuration")
    
    args = parser.parse_args()
    
    # Get the Farmer root directory
    script_dir = Path(__file__).parent
    farmer_root = script_dir.parent
    cert_dir = farmer_root / args.cert_dir
    
    print(f"Setting up HTTPS in {args.mode} mode...")
    print(f"Certificate directory: {cert_dir}")
    
    if args.mode == "dev":
        create_self_signed_cert(cert_dir, args.domain)
    elif args.mode == "prod":
        if not args.email:
            print("Error: --email is required for production mode")
            sys.exit(1)
        setup_lets_encrypt(args.domain, args.email, cert_dir)
    
    if args.nginx:
        create_nginx_config(args.domain, args.hr_port, args.it_port, cert_dir)
    
    print("\n✓ HTTPS setup complete!")
    print("\nTo start the assistants with HTTPS:")
    print(f"  cd {farmer_root}")
    print("  USE_HTTPS=1 python3 assistants/launcher/launch_all.py")
    
    if args.mode == "dev":
        print("\nNote: You'll need to accept the self-signed certificate warning in your browser.")
        print("For production, consider using a reverse proxy like nginx for better security.")

if __name__ == "__main__":
    main()
