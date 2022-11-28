import hivemind

# dht = hivemind.DHT(
#     host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
#     start=True)
#
# print('\n'.join(str(addr) for addr in dht.get_visible_maddrs()))
# print("Global IP:", hivemind.utils.networking.choose_ip_address(dht.get_visible_maddrs()))


import hivemind
dht = hivemind.DHT(
    host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
    initial_peers=[
        "/ip4/192.168.0.171/tcp/40909/p2p/QmRv5nREc5pYGGdGss7GigmvdMFMhoZHfZ81BUvFnvGS6F",
        "/ip4/192.168.0.171/udp/56731/quic/p2p/QmRv5nREc5pYGGdGss7GigmvdMFMhoZHfZ81BUvFnvGS6F",
    ], start=True)