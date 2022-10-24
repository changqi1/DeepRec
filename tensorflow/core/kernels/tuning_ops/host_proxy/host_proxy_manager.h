#ifndef __HOST_PROXY_MANAGER_H__
#define __HOST_PROXY_MANAGER_H__

#include "host_proxy.h"
#include <map>
#include <string>

typedef std::uintptr_t PROXY_HANDLE;

class HostOSTProxyManager {
  private:
    HostOSTProxyManager(/* args */) = default;
    virtual ~HostOSTProxyManager() = default;

    std::map<PROXY_HANDLE, HostProxy::Ptr> proxy_list;

  public:
    PROXY_HANDLE CreateNewProxy(const char *name);

    HostProxy::Ptr GetProxy(PROXY_HANDLE handle);

    void ReleaseProxy(PROXY_HANDLE handle);

    static HostOSTProxyManager &Instance() {
        static HostOSTProxyManager manager;
        return manager;
    }

    HostOSTProxyManager(HostOSTProxyManager &) = delete;
    HostOSTProxyManager(HostOSTProxyManager &&) = delete;
    void operator=(const HostOSTProxyManager &) = delete;
};

#endif