module.exports = {
  apps : [{
    name   : "pm2-mainnet-predictoor",
    script : "./pdr_backend/predictoor/main.py",
    args : "4",
    env: {
      PRIVATE_KEY : "be0712fe7ba53118a491218f8edd1460bb6b81b3eb1b55a4127575ffe6f1c3db",
      ADDRESS_FILE : "${HOME}/.ocean/ocean-contracts/artifacts/address.json",
      PAIR_FILTER : "BTC/USDT",
      TIMEFRAME_FILTER : "5m",
      SOURCE_FILTER : "binance",
      RPC_URL : "https://sapphire.oasis.io",
      SUBGRAPH_URL : "https://v4.subgraph.sapphire-mainnet.oceanprotocol.com/subgraphs/name/oceanprotocol/ocean-subgraph",
      STAKE_TOKEN : "0x39d22B78A7651A76Ffbde2aaAB5FD92666Aca520",
      OWNER_ADDRS : "0x4ac2e51f9b1b0ca9e000dfe6032b24639b172703"
    }
  }]
}
