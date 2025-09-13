from starlette.responses import JSONResponse, HTMLResponse
from starlette.requests import Request
from apps.mcp_server.simple_token_verifier import SimpleTokenVerifier, cost_extra_credit, DOMAIN


def home_page(request: Request) -> HTMLResponse:
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{DOMAIN} - Financial Tools MCP Server | Real-time Market Data & Analysis</title>
            <meta name="description" content="Access comprehensive financial tools including real-time stock data, options analysis, price predictions, market sentiment, and economic indicators through our MCP server API.">
            <meta name="keywords" content="financial tools, stock market API, options analysis, price prediction, market sentiment, economic indicators, yfinance, MCP server, financial data, investment tools">
            <meta name="author" content="{DOMAIN}">
            <meta name="robots" content="index, follow">
            
            <!-- Open Graph Meta Tags -->
            <meta property="og:title" content="{DOMAIN} - Financial Tools MCP Server">
            <meta property="og:description" content="Comprehensive financial tools including real-time stock data, options analysis, price predictions, and market sentiment indicators.">
            <meta property="og:type" content="website">
            <meta property="og:url" content="https://{DOMAIN}">
            <meta property="og:site_name" content="{DOMAIN}">
            
            <!-- Twitter Card Meta Tags -->
            <meta name="twitter:card" content="summary_large_image">
            <meta name="twitter:title" content="{DOMAIN} - Financial Tools MCP Server">
            <meta name="twitter:description" content="Access real-time financial data, options analysis, and market predictions through our MCP server API.">
            
            <!-- Canonical URL -->
            <link rel="canonical" href="https://{DOMAIN}">
            
            <!-- Structured Data (JSON-LD) -->
            <script type="application/ld+json">
            {{
                "@context": "https://schema.org",
                "@type": "WebSite",
                "name": "{DOMAIN} Financial Tools MCP Server",
                "description": "Comprehensive financial tools and APIs for real-time market data, options analysis, and price predictions",
                "url": "https://{DOMAIN}",
                "publisher": {{
                    "@type": "Organization",
                    "name": "{DOMAIN}",
                    "url": "https://{DOMAIN}"
                }},
                "mainEntity": {{
                    "@type": "SoftwareApplication",
                    "name": "Financial Tools MCP Server",
                    "applicationCategory": "FinanceApplication",
                    "operatingSystem": "Web",
                    "offers": {{
                        "@type": "Offer",
                        "price": "0",
                        "priceCurrency": "USD",
                        "availability": "https://schema.org/InStock"
                    }},
                    "featureList": [
                        "Real-time stock market data",
                        "Options chain analysis",
                        "Price prediction algorithms",
                        "Market sentiment indicators",
                        "Economic data integration",
                        "Financial statement analysis",
                        "Insider trading tracking",
                        "Institutional holdings data"
                    ]
                }}
            }}
            </script>
            
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f8f9fa;
                }}
                .container {{
                    background: white;
                    padding: 40px;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    font-size: 2.5em;
                    margin-bottom: 20px;
                }}
                h2 {{
                    color: #34495e;
                    font-size: 1.8em;
                    margin-top: 30px;
                    margin-bottom: 15px;
                }}
                h3 {{
                    color: #7f8c8d;
                    font-size: 1.3em;
                    margin-top: 20px;
                    margin-bottom: 10px;
                }}
                .feature-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }}
                .feature-card {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 6px;
                    border-left: 4px solid #3498db;
                }}
                .api-endpoint {{
                    background: #2c3e50;
                    color: white;
                    padding: 3px 8px;
                    border-radius: 4px;
                    font-family: 'Courier New', monospace;
                    font-size: 0.9em;
                }}
                .credit-cost {{
                    color: #e74c3c;
                    font-weight: bold;
                }}
                ul {{
                    padding-left: 20px;
                }}
                li {{
                    margin-bottom: 8px;
                }}
                a {{
                    color: #3498db;
                    text-decoration: none;
                }}
                a:hover {{
                    text-decoration: underline;
                }}
                .cta-section {{
                    background: #ecf0f1;
                    padding: 30px;
                    border-radius: 8px;
                    text-align: center;
                    margin-top: 40px;
                }}
                .btn {{
                    display: inline-block;
                    background: #3498db;
                    color: white;
                    padding: 12px 24px;
                    border-radius: 6px;
                    text-decoration: none;
                    font-weight: bold;
                    margin: 10px;
                }}
                .btn:hover {{
                    background: #2980b9;
                    text-decoration: none;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{DOMAIN} Financial Tools MCP Server</h1>
                <p><strong>Comprehensive financial data and analysis tools powered by Model Context Protocol (MCP).</strong></p>
                
                <h2>🚀 Available Financial Services</h2>
                <p>Our MCP server provides real-time access to financial markets, economic data, and advanced analytics tools. Perfect for developers, traders, and financial analysts.</p>
                
                <div class="feature-grid">
                    <div class="feature-card">
                        <h3>📈 Market Data & Analysis</h3>
                        <ul>
                            <li><span class="api-endpoint">get_ticker_data</span> - Real-time stock information <span class="credit-cost">(1 credit)</span></li>
                            <li><span class="api-endpoint">get_price_history</span> - Historical price data <span class="credit-cost">(2 credits)</span></li>
                            <li><span class="api-endpoint">get_financial_statements</span> - Company financials <span class="credit-cost">(1 credit)</span></li>
                            <li><span class="api-endpoint">get_earnings_history</span> - Earnings calendar <span class="credit-cost">(1 credit)</span></li>
                            <li><span class="api-endpoint">get_ticker_news</span> - Latest market news <span class="credit-cost">(1 credit)</span></li>
                        </ul>
                    </div>
                    
                    <div class="feature-card">
                        <h3>🔮 Predictive Analytics</h3>
                        <ul>
                            <li><span class="api-endpoint">price_prediction</span> - ML-powered price forecasts <span class="credit-cost">(3 credits)</span></li>
                            <li>Advanced XGBoost models trained on technical indicators</li>
                            <li>Pattern recognition and trend analysis</li>
                            <li>Risk assessment and probability scoring</li>
                        </ul>
                    </div>
                    
                    <div class="feature-card">
                        <h3>📊 Options Trading Tools</h3>
                        <ul>
                            <li><span class="api-endpoint">super_option_tool</span> - Comprehensive options analysis <span class="credit-cost">(2 credits)</span></li>
                            <li>Options chain data and Greeks calculation</li>
                            <li>Implied volatility analysis</li>
                            <li>Options flow and unusual activity detection</li>
                        </ul>
                    </div>
                    
                    <div class="feature-card">
                        <h3>🏢 Institutional Intelligence</h3>
                        <ul>
                            <li><span class="api-endpoint">get_top25_holders</span> - Major shareholders <span class="credit-cost">(1 credit)</span></li>
                            <li><span class="api-endpoint">get_insider_trades</span> - Insider trading activity <span class="credit-cost">(1 credit)</span></li>
                            <li>13F filings and institutional ownership changes</li>
                            <li>Smart money tracking and analysis</li>
                        </ul>
                    </div>
                    
                    <div class="feature-card">
                        <h3>📰 Market Sentiment</h3>
                        <ul>
                            <li><span class="api-endpoint">get_overall_sentiment</span> - CNN Fear & Greed Index <span class="credit-cost">(1 credit)</span></li>
                            <li>Market emotion indicators</li>
                            <li>Social media sentiment analysis</li>
                            <li>News sentiment scoring</li>
                        </ul>
                    </div>
                    
                    <div class="feature-card">
                        <h3>🌍 Economic Data</h3>
                        <ul>
                            <li><span class="api-endpoint">get_fred_series</span> - Federal Reserve Economic Data <span class="credit-cost">(1 credit)</span></li>
                            <li><span class="api-endpoint">search_fred_series</span> - FRED database search <span class="credit-cost">(1 credit)</span></li>
                            <li><span class="api-endpoint">cnbc_news_feed</span> - Financial news feed <span class="credit-cost">(2 credits)</span></li>
                            <li><span class="api-endpoint">social_media_feed</span> - Social sentiment data <span class="credit-cost">(2 credits)</span></li>
                        </ul>
                    </div>
                </div>
                
                <h2>🔧 Utility Tools</h2>
                <ul>
                    <li><span class="api-endpoint">calculate</span> - Advanced financial calculations (Free)</li>
                    <li><span class="api-endpoint">get_current_time</span> - Market hours and time zones (Free)</li>
                </ul>
                
                <h2>💡 Key Features</h2>
                <ul>
                    <li><strong>Real-time Data:</strong> Live market feeds from premium sources</li>
                    <li><strong>Credit-based System:</strong> Flexible pricing with per-request billing</li>
                    <li><strong>Authentication:</strong> Secure API key management with Unkey integration</li>
                    <li><strong>Scalable:</strong> Built on FastMCP for high-performance API delivery</li>
                    <li><strong>Developer Friendly:</strong> RESTful endpoints with comprehensive error handling</li>
                    <li><strong>AI Integration:</strong> Optimized for LLM consumption and tool calling</li>
                </ul>
                
                <div class="cta-section">
                    <h2>🚀 Get Started Today</h2>
                    <p>Start building powerful financial applications with our comprehensive API suite.</p>
                    <a href="https://{DOMAIN}/docs" class="btn">📚 View Documentation</a>
                    <a href="https://{DOMAIN}/api-keys" class="btn">🔑 Get API Key</a>
                    <a href="https://{DOMAIN}/pricing" class="btn">💰 Pricing Plans</a>
                </div>
                
                <h2>🔗 Quick Links</h2>
                <ul>
                    <li><a href="https://{DOMAIN}/docs">API Documentation</a> - Complete endpoint reference and examples</li>
                    <li><a href="https://{DOMAIN}/status">Service Status</a> - Real-time system health and uptime</li>
                    <li><a href="https://{DOMAIN}/support">Support</a> - Get help and troubleshooting assistance</li>
                    <li><a href="https://{DOMAIN}/blog">Blog</a> - Latest updates and financial insights</li>
                </ul>
                
                <p><small>© 2024 {DOMAIN}. All rights reserved. | <a href="https://{DOMAIN}/privacy">Privacy Policy</a> | <a href="https://{DOMAIN}/terms">Terms of Service</a></small></p>
            </div>
        </body>
    </html>
    """)
