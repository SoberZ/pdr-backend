module.exports = {
  apps : [{
    name   : "pm2-testnet-predictoor",
    script : "./pdr_backend/predictoor/main.py",
    args : "4",
    env: {
      PRIVATE_KEY : "be0712fe7ba53118a491218f8edd1460bb6b81b3eb1b55a4127575ffe6f1c3db",
      ADDRESS_FILE : "${HOME}/.ocean/oceans-contracts/artifacts/address.json",
      PAIR_FILTER : "BTC/USDT",
      TIMEFRAME_FILTER : "5m",
      SOURCE_FILTER : "binance",
      RPC_URL : "https://testnet.sapphire.oasis.dev",
      SUBGRAPH_URL : "https://v4.subgraph.sapphire-testnet.oceanprotocol.com/subgraphs/name/oceanprotocol/ocean-subgraph",
      STAKE_TOKEN : "0x973e69303259B0c2543a38665122b773D28405fB",
      OWNER_ADDRS : "0xe02a421dfc549336d47efee85699bd0a3da7d6ff"
    }
  }]
}
