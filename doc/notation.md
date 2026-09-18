# Notation

This notation applies to `data/crsp_dsf_2024.csv.gz`, the CRSP Daily Stock File
covering trading dates from 2024-01-02 through 2024-12-31.

| Symbol | Definition | CRSP field / formula |
| --- | --- | --- |
| \(i\) | Security identifier | `permno` (CRSP PERMNO) |
| \(t\) | Trading date | `date` |
| \(p_{i,t}\) | Closing price | \(\lvert\texttt{prc}_{i,t}\rvert\). CRSP may report a negative `prc` when the price is bid/ask-derived. |
| \(r_{i,t}\) | Daily total return | `ret`; includes distributions when available. |
| \(r^x_{i,t}\) | Daily return excluding distributions | `retx`. |
| \(V_{i,t}\) | Trading volume | `vol`, the number of shares traded on date \(t\). |
| \(S_{i,t}\) | Shares outstanding | \(1{,}000 \times \texttt{shrout}_{i,t}\), since `shrout` is reported in thousands of shares. |
| \(DV_{i,t}\) | Dollar volume | \(p_{i,t} \times V_{i,t}\). |
| \(\tau_{i,t}\) | Share turnover | \(V_{i,t}/S_{i,t}\). |
| \(\sigma_{i,t}^{(21)}\) | 21-trading-day realized volatility | Sample standard deviation of the available daily \(r_{i,u}\) for \(u=t-20,\ldots,t\). Annualized volatility is \(\sigma_{i,t}^{(21)}\sqrt{252}\). |
| \(b_{i,t}\), \(a_{i,t}\) | Bid and ask prices | `bid`, `ask` |
| \(h_{i,t}\), \(l_{i,t}\) | Daily high and low prices | `askhi`, `bidlo` |
| \(o_{i,t}\) | Opening price | `openprc` |
| \(N_{i,t}\) | Number of reported trades | `numtrd` |

Use `ret` for total-return analysis and `retx` only when an analysis explicitly
excludes distributions. Observations with missing or non-finite inputs are
excluded before calculating derived measures.
