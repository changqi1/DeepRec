#include "host_proxy_manager.h"

PROXY_HANDLE HostOSTProxyManager::CreateNewProxy(const char *name) {
    if (name == nullptr) {
        std::string msg = "A name must be specified!";
        // throw std::runtime_error(msg); todo(marvin): show the error info within tf.
    }

    auto proxy = std::make_shared<HostProxy>(name);
    auto handle = reinterpret_cast<PROXY_HANDLE>(proxy.get());
    proxy_list[handle] = proxy;
    return handle;
}

HostProxy::Ptr HostOSTProxyManager::GetProxy(PROXY_HANDLE handle) {
    auto search = proxy_list.find(handle);
    if (search == proxy_list.end())
        return nullptr;

    return search->second;
}

void HostOSTProxyManager::ReleaseProxy(PROXY_HANDLE handle) {
    auto search = proxy_list.find(handle);
    if (search == proxy_list.end())
        return;

    proxy_list.erase(handle);
}