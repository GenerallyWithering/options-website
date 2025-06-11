# Options Strategy Simulator – Streamlit App

[![Python](https://img.shields.io/badge/python-3.8%2B-blue?logo=python&style=flat-square)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.0-orange?logo=streamlit&style=flat-square)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/plotly-5.0-blue?logo=plotly&style=flat-square)](https://plotly.com/)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

---

## Overview  
This is an advanced options strategy simulator built with Streamlit. It enables users to construct, analyze, and share multi-leg options strategies through rich visualizations and detailed analytics. Designed for traders, educators, and learners, the app focuses on flexibility, interactivity, and financial clarity.

---

## Current Features

### User Authentication & Session  
- Secure login using `streamlit-authenticator`  
- Password hashing with `passlib`  
- Persistent login with "Remember Me" functionality  
- Logging of login attempts for auditing  

### Strategy Builder  
- Add multiple legs: calls, puts, long/short, any strike and expiry  
- Supports common strategies like Straddles, Strangles, Iron Condors, Butterflies, and custom legs  
- Automatic detection and display of net debit or credit  
- Calculates maximum profit, maximum loss, and breakeven points  
- Export strategy details as CSV  

### Payoff Chart & Analysis  
- Interactive payoff chart using Plotly with dynamic zoom  
- Ability to toggle visibility of individual legs  
- Highlighted breakeven points and profit/loss zones  
- Live updates as strategy legs are edited  

### Greeks Calculator  
- Real-time Delta, Gamma, Theta, Vega, and Rho per leg and total  
- Displayed in an organized table  
- Greeks over time visualization (in progress)  

### Additional Features  
- Clean and responsive layout  
- Modular codebase with database integration  
- Admin-controlled database initialization  
- Basic glossary and user notes functionality  

---

## Future Features (Planned)

### User Experience  
- Homepage with clear navigation  
- Email validation during registration  
- Dark mode and theme toggles  
- Educational tooltips and modals  
- Snapshot & share feature with watermark  

### Strategy Enhancements  
- Save/load user strategies tied to profiles  
- Strategy notebook for journaling reasoning and details  
- Tagging system (e.g., bullish, earnings)  
- Public sharing via unique URLs  
- Comments and ratings on shared strategies  
- Watchlist and quick-build from favorites  
- Multi-account paper portfolios  

### Advanced Visualization & Analytics  
- 3D payoff surface (stock price vs. time vs. P/L)  
- P/L heatmap for returns over time and price  
- Theta decay waterfall visualization  
- Simplified backtesting engine  
- Probability of Profit (PoP) estimator  
- Historical strategy replay  

### Market Data & Broker Integrations  
- Real-time bid/ask, implied volatility, delta, and volume from APIs  
- Broker API integration for paper or real trading (e.g., Alpaca, TastyTrade)  

---

## Roadmap

### Phase 1 – Essentials  
- Fix margin requirement calculations  
- Implement full CSV download functionality  
- Develop responsive homepage with overview  
- Add email format verification  

### Phase 2 – Strategy Management  
- User strategy saving and loading  
- Strategy history logbook  
- Notes and comments per strategy  
- Public sharing via unique IDs  

### Phase 3 – Education & Glossary  
- Visual explainers for common strategies  
- Expanded glossary of terms  
- Informative modals for key concepts  

### Phase 4 – Advanced Visuals  
- Integration of 3D payoff surface  
- P/L heatmap and theta decay animations  
- Basic backtesting simulation  

### Phase 5 – Social & Sharing  
- Public strategy browsing  
- Comments and star ratings on strategies  
- Snapshot sharing with watermark  

### Phase 6 – Pro-Level Tools  
- Probability of Profit calculator  
- Historical strategy replay  
- Multi-account paper trading dashboard  

### Phase 7 – Live Data & Trading  
- Live market data feed integration  
- Broker login for real or simulated trading  
- Real-time Greeks and bid/ask feeds  

---

## Built With  
- [Streamlit](https://streamlit.io/)  
- [Plotly](https://plotly.com/)  
- [Passlib](https://passlib.readthedocs.io/)  
- [streamlit-authenticator](https://github.com/mkhorasani/Streamlit-Authenticator)  
- Python, Numpy, Pandas  

---

