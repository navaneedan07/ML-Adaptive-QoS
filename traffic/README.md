# Traffic Feature Pipeline

Converts raw PCAP files into a labeled `features.csv` for ML training.
Fits into the team repo structure as `traffic/capture.py`,
`traffic/flow_builder.py`, `traffic/feature_extractor.py`.

## Pipeline

```
PCAP files  --(capture.py)-->  packets
            --(flow_builder.py)-->  flows (bidirectional 5-tuple)
            --(feature_extractor.py)-->  features.csv
```

Note: `capture.py` here reads *existing* PCAP files (offline parsing),
matching the team's naming convention. It does not do live packet
capture — if the team needs that later, it can be added as a separate
function in this same file.

## Setup

```bash
pip install scapy
```

## Dataset

Using **ISCX VPN-nonVPN 2016**: https://www.unb.ca/cic/datasets/vpn.html

Download the PCAP files and place them in `../data/raw/` (one `.pcap`
per application/activity, as distributed by UNB, e.g.
`skype_chat1a.pcap`, `facebook_video1a.pcap`, `vpn_email2a.pcap`).
Processed output (`features.csv`) goes in `../data/processed/`.

## Run

```bash
cd traffic
python feature_extractor.py ../data/raw ../data/processed/features.csv
```

This parses every `.pcap` in `../data/raw/`, groups packets into flows,
computes features, and writes one row per flow to `features.csv` in
`../data/processed/`.

## Output columns (features.csv)

| Column | Meaning |
|---|---|
| `src_ip`, `src_port`, `dst_ip`, `dst_port`, `proto` | Flow identifier |
| `flow_duration` | Seconds from first to last packet in the flow |
| `packet_count` | Number of packets in the flow |
| `total_bytes` | Sum of packet sizes |
| `mean_packet_size`, `std_packet_size` | Packet size stats |
| `mean_iat`, `std_iat` | Inter-arrival time stats (seconds) |
| `num_bursts`, `mean_burst_size` | Burst statistics (gap > 1.0s starts a new burst) |
| `is_vpn` | 1 if filename indicates VPN traffic, else 0 |
| `source_file` | Originating pcap filename |
| `label` | Traffic class guessed from filename (chat / email / streaming / voip / p2p / file_transfer / browsing / unknown) |

## Important: check the `label` column

`label_from_filename()` in `feature_extractor.py` guesses the class from
keywords in each pcap's filename. **Once you've downloaded the real ISCX
files, print `features.csv` and check for `unknown` labels** — extend
`LABEL_KEYWORDS` in `feature_extractor.py` to match the exact filenames
you get. This is the "F. Validate the dataset" step in the plan: also
check for missing values, duplicate flows, class imbalance, and
suspiciously out-of-range feature values before handing this off.

## For Hitesh

`features.csv` is the ML-ready input — one row per flow, `label` is the
target column, everything else is a feature (drop `src_ip`/`dst_ip`/
`source_file` before training; keep `proto`/`is_vpn` as categorical
features if useful).
