// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <httplib.h>
#include "common/logging/log.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/libs.h"
#include "core/libraries/network/http.h"
#include "http_error.h"

namespace Libraries::Http {

static bool g_isHttpInitialized = true; // TODO temp always inited

// HTTP state machinery
namespace {

constexpr int kHttpConnectTimeoutSec = 5;
constexpr int kHttpReadTimeoutSec = 30;

struct HttpTemplate {
    int libCtxId;
    std::string userAgent;
    int httpVerMajor = 1;
    int httpVerMinor = 1;
    bool autoRedirect = true;
    uint32_t connectTimeoutUs = 0;
    uint32_t recvTimeoutUs = 0;
    uint32_t sendTimeoutUs = 0;
    uint32_t resolveTimeoutUs = 0;

    bool acceptEncodingGzip = true;
    bool inflateGzip = true;
    bool chunkedTransfer = false;
    bool nonblock = false;
    bool authEnabled = false;
    bool cookieEnabled = false;
    int recvBlockSize = 0;         // 0 = httplib default
    int responseHeaderMaxSize = 0; // 0 = no cap
    uint32_t resolveRetry = 0;     // store-and-forget
    int sslMinVersion = 0;         // 0 = library default
    uint32_t sslFlags = 0;         // ORBIS_HTTPS_FLAG_*
    // proxy
    std::string proxyHost;
    int proxyPort = 0;
    int proxyMode = 0; // ORBIS_HTTP_PROXY_AUTO=0, MANUAL=1
    int proxyVersion = 0;
};

struct HttpConnection {
    int templateId;
    std::string scheme; // "http" or "https"
    std::string host;
    int port;
    bool keepAlive;
    std::unique_ptr<httplib::Client> client;

    // Connection-level overrides of template settings.
    bool autoRedirect = true;
    bool nonblock = false;
    bool cookieEnabled = false;
    bool authEnabled = false;
    uint32_t connectTimeoutUs = 0;
    uint32_t recvTimeoutUs = 0;
    uint32_t sendTimeoutUs = 0;
    uint32_t resolveTimeoutUs = 0;
};

struct HttpRequest {
    int connectionId;
    s32 method;
    std::string path; // path + query, no scheme/host
    u64 contentLength = 0;
    std::vector<std::pair<std::string, std::string>> requestHeaders;

    // Cancellation flag set by sceHttpAbortRequest
    bool aborted = false;

    // Filled by sceHttpSendRequest.
    bool sent = false;
    int statusCode = 0;
    std::string responseBody;
    size_t readOffset = 0;
    std::string responseHeadersBlob;
};

struct StoredCookie {
    std::string url; // domain context the cookie applies to
    std::string raw; // full "name=value; Path=/; Expires=..." string
};
std::vector<StoredCookie> g_cookies;

std::mutex g_state_mutex;
std::map<int, HttpTemplate> g_templates;
std::map<int, HttpConnection> g_connections;
std::map<int, HttpRequest> g_requests;
int g_next_template_id = 1;
int g_next_connection_id = 1;
int g_next_request_id = 1;

const char* MethodName(s32 method) {
    switch (method) {
    case 0: // SCE_HTTP_METHOD_GET
        return "GET";
    case 1: // SCE_HTTP_METHOD_POST
        return "POST";
    case 2: // SCE_HTTP_METHOD_HEAD
        return "HEAD";
    case 3: // SCE_HTTP_METHOD_OPTIONS
        return "OPTIONS";
    case 4: // SCE_HTTP_METHOD_PUT
        return "PUT";
    case 5: // SCE_HTTP_METHOD_DELETE
        return "DELETE";
    case 6: // SCE_HTTP_METHOD_TRACE
        return "TRACE";
    case 7: // SCE_HTTP_METHOD_CONNECT
        return "CONNECT";
    case 8: // out-of-band PATCH (SceHttpMethods doesn't define it)
        return "PATCH";
    default:
        return nullptr;
    }
}

// Minimal URL parser. Splits "scheme://host[:port][/path]" into parts.
struct ParsedUrl {
    std::string scheme;
    std::string host;
    int port = 0;
    std::string path; // includes leading '/' and any '?query'
};

bool ParseUrl(const std::string& url, ParsedUrl& out) {
    const auto schemeEnd = url.find("://");
    if (schemeEnd == std::string::npos) {
        return false;
    }
    out.scheme = url.substr(0, schemeEnd);
    if (out.scheme != "http" && out.scheme != "https") {
        return false;
    }
    const auto hostStart = schemeEnd + 3;
    const auto pathStart = url.find('/', hostStart);
    std::string hostPort;
    if (pathStart == std::string::npos) {
        hostPort = url.substr(hostStart);
        out.path = "/";
    } else {
        hostPort = url.substr(hostStart, pathStart - hostStart);
        out.path = url.substr(pathStart);
    }
    if (hostPort.empty()) {
        return false;
    }
    const auto colonPos = hostPort.find(':');
    if (colonPos == std::string::npos) {
        out.host = hostPort;
        out.port = (out.scheme == "https") ? 443 : 80;
    } else {
        out.host = hostPort.substr(0, colonPos);
        try {
            out.port = std::stoi(hostPort.substr(colonPos + 1));
        } catch (...) {
            return false;
        }
        if (out.port <= 0 || out.port > 65535) {
            return false;
        }
    }
    return true;
}

// Build a "Name: Value\r\nName: Value\r\n" blob from cpp-httplib headers.
std::string SerializeHeadersBlob(const httplib::Headers& headers) {
    std::string out;
    for (const auto& [name, value] : headers) {
        out.append(name);
        out.append(": ");
        out.append(value);
        out.append("\r\n");
    }
    return out;
}

void ApplyClientTimeouts(httplib::Client& cli, const HttpConnection& conn,
                         const HttpTemplate* tmpl) {
    auto resolveUs = [&](uint32_t connVal, uint32_t tmplVal, int defaultSec) {
        if (connVal != 0)
            return std::chrono::microseconds(connVal);
        if (tmplVal != 0)
            return std::chrono::microseconds(tmplVal);
        return std::chrono::microseconds(defaultSec * 1'000'000ll);
    };

    cli.set_connection_timeout(resolveUs(conn.connectTimeoutUs, tmpl ? tmpl->connectTimeoutUs : 0,
                                         kHttpConnectTimeoutSec));
    cli.set_read_timeout(
        resolveUs(conn.recvTimeoutUs, tmpl ? tmpl->recvTimeoutUs : 0, kHttpReadTimeoutSec));
    cli.set_write_timeout(
        resolveUs(conn.sendTimeoutUs, tmpl ? tmpl->sendTimeoutUs : 0, kHttpReadTimeoutSec));
}

httplib::Client* GetOrCreateClient(HttpConnection& conn) {
    if (!conn.client) {
        if (conn.scheme == "https") {
            LOG_ERROR(Lib_Http,
                      "HTTPS requested but SSL client not compiled in; "
                      "falling back to plain HTTP for {}:{}",
                      conn.host, conn.port);
        }
        conn.client = std::make_unique<httplib::Client>(conn.host, conn.port);
        const auto tmplIt = g_templates.find(conn.templateId);
        const HttpTemplate* tmpl = (tmplIt != g_templates.end()) ? &tmplIt->second : nullptr;
        ApplyClientTimeouts(*conn.client, conn, tmpl);
        conn.client->set_keep_alive(conn.keepAlive);
        conn.client->set_follow_location(conn.autoRedirect);
        if (tmpl && !tmpl->userAgent.empty()) {
            conn.client->set_default_headers({{"User-Agent", tmpl->userAgent}});
        }
    }
    return conn.client.get();
}

enum class SettingScope { None, Template, Connection, Request };

struct SettingTarget {
    SettingScope scope = SettingScope::None;
    HttpTemplate* tmpl = nullptr;
    HttpConnection* conn = nullptr;
    HttpRequest* req = nullptr;
};

SettingTarget ResolveSettingsId(int id) {
    SettingTarget out;
    auto tIt = g_templates.find(id);
    if (tIt != g_templates.end()) {
        out.scope = SettingScope::Template;
        out.tmpl = &tIt->second;
        return out;
    }
    auto cIt = g_connections.find(id);
    if (cIt != g_connections.end()) {
        out.scope = SettingScope::Connection;
        out.conn = &cIt->second;
        return out;
    }
    auto rIt = g_requests.find(id);
    if (rIt != g_requests.end()) {
        out.scope = SettingScope::Request;
        out.req = &rIt->second;
        return out;
    }
    return out;
}

} // namespace

void NormalizeAndAppendPath(char* dest, char* src) {
    char* lastSlash;
    u64 length;

    lastSlash = strrchr(dest, '/');
    if (lastSlash == NULL) {
        length = strlen(dest);
        dest[length] = '/';
        dest[length + 1] = '\0';
    } else {
        lastSlash[1] = '\0';
    }
    if (*src == '/') {
        dest[0] = '\0';
    }
    length = strnlen(dest, 0x3fff);
    strncat(dest, src, 0x3fff - length);
    return;
}

int HttpRequestInternal_Acquire(HttpRequestInternal** outRequest, u32 requestId) {
    return 0; // TODO dummy
}
int HttpRequestInternal_Release(HttpRequestInternal* request) {
    return 0; // TODO dummy
}

int PS4_SYSV_ABI sceHttpAbortRequest(int reqId) {
    std::scoped_lock lk{g_state_mutex};
    auto it = g_requests.find(reqId);
    if (it == g_requests.end()) {
        return ORBIS_HTTP_ERROR_INVALID_ID;
    }
    it->second.aborted = true;
    LOG_DEBUG(Lib_Http, "sceHttpAbortRequest: req={} flagged for abort", reqId);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpAbortRequestForce(int reqId) {
    return sceHttpAbortRequest(reqId);
}

int PS4_SYSV_ABI sceHttpAbortWaitRequest(int reqId) {
    return sceHttpAbortRequest(reqId);
}

int PS4_SYSV_ABI sceHttpAddCookie(int libhttpCtxId, const char* url, const char* cookie,
                                  u64 cookieLength) {
    if (url == nullptr || cookie == nullptr) {
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lk{g_state_mutex};
    StoredCookie c;
    c.url.assign(url);
    c.raw.assign(cookie, cookie + cookieLength);
    g_cookies.push_back(std::move(c));
    LOG_DEBUG(Lib_Http, "sceHttpAddCookie: stored cookie for url={} ({} chars)", url, cookieLength);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpAddQuery(int reqId, const char* name, const char* value) {
    if (name == nullptr || value == nullptr) {
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lk{g_state_mutex};
    auto it = g_requests.find(reqId);
    if (it == g_requests.end()) {
        return ORBIS_HTTP_ERROR_INVALID_ID;
    }
    HttpRequest& req = it->second;
    // Append "name=value" to the query portion of the path. Leading
    // separator is ? if no query exists, & otherwise.
    const bool hasQuery = req.path.find('?') != std::string::npos;
    req.path += hasQuery ? '&' : '?';
    req.path += name;
    req.path += '=';
    req.path += value;
    LOG_DEBUG(Lib_Http, "sceHttpAddQuery: req={} {}={} -> path={}", reqId, name, value, req.path);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpAddRequestHeader(int id, const char* name, const char* value, s32 mode) {
    if (name == nullptr || value == nullptr) {
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lk{g_state_mutex};
    auto it = g_requests.find(id);
    if (it == g_requests.end()) {
        LOG_WARNING(Lib_Http, "sceHttpAddRequestHeader: req={} not found", id);
        return ORBIS_HTTP_ERROR_INVALID_ID;
    }
    // mode: 0 = ADD (append), 1 = OVERWRITE (replace existing).
    HttpRequest& req = it->second;
    if (mode == 1) {
        for (auto h = req.requestHeaders.begin(); h != req.requestHeaders.end();) {
            if (h->first == name) {
                h = req.requestHeaders.erase(h);
            } else {
                ++h;
            }
        }
    }
    req.requestHeaders.emplace_back(name, value);
    LOG_DEBUG(Lib_Http, "sceHttpAddRequestHeader: req={} '{}: {}' (mode={})", id, name, value,
              mode);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpAddRequestHeaderRaw(int reqId, const char* raw, u64 rawLen) {
    if (raw == nullptr || rawLen == 0) {
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }
    std::string s(raw, rawLen);
    while (!s.empty() && (s.back() == '\r' || s.back() == '\n'))
        s.pop_back();
    const auto colon = s.find(':');
    if (colon == std::string::npos) {
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }
    std::string name = s.substr(0, colon);
    std::string value = s.substr(colon + 1);
    while (!value.empty() && value.front() == ' ')
        value.erase(value.begin());
    return sceHttpAddRequestHeader(reqId, name.c_str(), value.c_str(),
                                   /*mode=*/1 /*overwrite*/);
}

int PS4_SYSV_ABI sceHttpAuthCacheExport(int libhttpCtxId, void* buffer, u64 bufferSize,
                                        u64* exportSize) {
    if (exportSize) {
        *exportSize = 0;
    }
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpAuthCacheFlush(int libhttpCtxId) {
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpAuthCacheImport(int libhttpCtxId, const void* buffer, u64 bufferSize) {
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpCacheRedirectedConnectionEnabled(int id, int isEnable) {
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpCookieExport(int libhttpCtxId, void* buffer, u64 bufferSize,
                                     u64* exportSize) {
    std::scoped_lock lk{g_state_mutex};
    u64 total = 0;
    for (const auto& c : g_cookies) {
        total += c.raw.size();
    }
    if (exportSize) {
        *exportSize = total;
    }
    if (buffer != nullptr && bufferSize >= total) {
        char* dst = static_cast<char*>(buffer);
        size_t off = 0;
        for (const auto& c : g_cookies) {
            std::memcpy(dst + off, c.raw.data(), c.raw.size());
            off += c.raw.size();
        }
    }
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpCookieFlush(int libhttpCtxId) {
    std::scoped_lock lk{g_state_mutex};
    g_cookies.clear();
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpCookieImport(int libhttpCtxId, const void* buffer, u64 bufferSize) {
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpCreateConnection(int tmplId, const char* host, const char* scheme, u16 port,
                                         bool enableKeepalive) {
    if (host == nullptr || scheme == nullptr) {
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }
    std::string url;
    url.reserve(strlen(scheme) + 3 + strlen(host) + 8);
    url.append(scheme);
    url.append("://");
    url.append(host);
    url.append(":");
    url.append(std::to_string(port));
    return sceHttpCreateConnectionWithURL(tmplId, url.c_str(), enableKeepalive);
}

int PS4_SYSV_ABI sceHttpCreateConnectionWithURL(int tmplId, const char* url, bool enableKeepalive) {
    if (url == nullptr) {
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }
    ParsedUrl parsed;
    if (!ParseUrl(url, parsed)) {
        LOG_ERROR(Lib_Http, "sceHttpCreateConnectionWithURL: invalid URL '{}'", url);
        return ORBIS_HTTP_ERROR_INVALID_URL;
    }

    std::scoped_lock lk{g_state_mutex};
    // np_web_api in doesn't always create a template before a connection (sceHttpInit's return
    // value gets re-used as a "context" id)
    if (g_templates.find(tmplId) == g_templates.end()) {
        LOG_DEBUG(Lib_Http,
                  "sceHttpCreateConnectionWithURL: tmplId={} not registered, "
                  "treating as default",
                  tmplId);
    }

    const int id = g_next_connection_id++;
    HttpConnection& conn = g_connections[id];
    conn.templateId = tmplId;
    conn.scheme = parsed.scheme;
    conn.host = parsed.host;
    conn.port = parsed.port;
    conn.keepAlive = enableKeepalive;
    // client constructed lazily on first send

    LOG_INFO(Lib_Http, "sceHttpCreateConnectionWithURL: tmpl={} url='{}' -> conn={} ({}://{}:{})",
             tmplId, url, id, parsed.scheme, parsed.host, parsed.port);
    return id;
}

int PS4_SYSV_ABI sceHttpCreateEpoll(int libhttpCtxId, void* eh) {
    return 1; // dummy
}

int PS4_SYSV_ABI sceHttpCreateRequest(int connId, s32 method, const char* path, u64 contentLength) {
    if (path == nullptr) {
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lk{g_state_mutex};
    auto connIt = g_connections.find(connId);
    if (connIt == g_connections.end()) {
        return ORBIS_HTTP_ERROR_INVALID_ID;
    }
    const int id = g_next_request_id++;
    HttpRequest& req = g_requests[id];
    req.connectionId = connId;
    req.method = method;
    req.path = path;
    req.contentLength = contentLength;
    LOG_INFO(Lib_Http, "sceHttpCreateRequest: conn={} method={} path='{}' contentLen={} -> req={}",
             connId, MethodName(method), path, contentLength, id);
    return id;
}

int PS4_SYSV_ABI sceHttpCreateRequest2(int connId, const char* method, const char* path,
                                       u64 contentLength) {
    if (path == nullptr || method == nullptr) {
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }
    s32 m = 0; // GET
    if (std::strcmp(method, "GET") == 0)
        m = 0;
    else if (std::strcmp(method, "POST") == 0)
        m = 1;
    else if (std::strcmp(method, "HEAD") == 0)
        m = 2;
    else if (std::strcmp(method, "OPTIONS") == 0)
        m = 3;
    else if (std::strcmp(method, "PUT") == 0)
        m = 4;
    else if (std::strcmp(method, "DELETE") == 0)
        m = 5;
    else if (std::strcmp(method, "TRACE") == 0)
        m = 6;
    else if (std::strcmp(method, "CONNECT") == 0)
        m = 7;
    else if (std::strcmp(method, "PATCH") == 0)
        m = 8; // out-of-band PATCH marker; see sendRequest
    else
        return ORBIS_HTTP_ERROR_UNKNOWN_METHOD;
    return sceHttpCreateRequest(connId, m, path, contentLength);
}

int PS4_SYSV_ABI sceHttpCreateRequestWithURL(int connId, s32 method, const char* url,
                                             u64 contentLength) {
    if (url == nullptr) {
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }
    ParsedUrl parsed;
    if (!ParseUrl(url, parsed)) {
        LOG_ERROR(Lib_Http, "sceHttpCreateRequestWithURL: invalid URL '{}'", url);
        return ORBIS_HTTP_ERROR_INVALID_URL;
    }

    std::scoped_lock lk{g_state_mutex};
    auto connIt = g_connections.find(connId);
    if (connIt == g_connections.end()) {
        LOG_ERROR(Lib_Http, "sceHttpCreateRequestWithURL: conn={} not found", connId);
        return ORBIS_HTTP_ERROR_INVALID_ID;
    }

    const int id = g_next_request_id++;
    HttpRequest& req = g_requests[id];
    req.connectionId = connId;
    req.method = method;
    req.path = parsed.path;
    req.contentLength = contentLength;

    LOG_INFO(Lib_Http,
             "sceHttpCreateRequestWithURL: conn={} method={} url='{}' contentLen={} -> req={}",
             connId, MethodName(method), url, contentLength, id);
    return id;
}

int PS4_SYSV_ABI sceHttpCreateRequestWithURL2(int connId, const char* method, const char* url,
                                              u64 contentLength) {
    if (url == nullptr || method == nullptr) {
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }
    s32 m;
    if (std::strcmp(method, "GET") == 0)
        m = 0;
    else if (std::strcmp(method, "POST") == 0)
        m = 1;
    else if (std::strcmp(method, "HEAD") == 0)
        m = 2;
    else if (std::strcmp(method, "OPTIONS") == 0)
        m = 3;
    else if (std::strcmp(method, "PUT") == 0)
        m = 4;
    else if (std::strcmp(method, "DELETE") == 0)
        m = 5;
    else if (std::strcmp(method, "TRACE") == 0)
        m = 6;
    else if (std::strcmp(method, "CONNECT") == 0)
        m = 7;
    else if (std::strcmp(method, "PATCH") == 0)
        m = 8; // out-of-band PATCH marker
    else
        return ORBIS_HTTP_ERROR_UNKNOWN_METHOD;
    return sceHttpCreateRequestWithURL(connId, m, url, contentLength);
}

int PS4_SYSV_ABI sceHttpCreateTemplate() {
    std::scoped_lock lk{g_state_mutex};
    const int id = g_next_template_id++;
    g_templates[id] = HttpTemplate{};
    LOG_DEBUG(Lib_Http, "sceHttpCreateTemplate -> {}", id);
    return id;
}

int PS4_SYSV_ABI sceHttpDbgEnableProfile() {
    return ORBIS_OK;
}
int PS4_SYSV_ABI sceHttpDbgGetConnectionStat() {
    return ORBIS_OK;
}
int PS4_SYSV_ABI sceHttpDbgGetRequestStat() {
    return ORBIS_OK;
}
int PS4_SYSV_ABI sceHttpDbgSetPrintf() {
    return ORBIS_OK;
}
int PS4_SYSV_ABI sceHttpDbgShowConnectionStat() {
    return ORBIS_OK;
}
int PS4_SYSV_ABI sceHttpDbgShowMemoryPoolStat() {
    return ORBIS_OK;
}
int PS4_SYSV_ABI sceHttpDbgShowRequestStat() {
    return ORBIS_OK;
}
int PS4_SYSV_ABI sceHttpDbgShowStat() {
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpDeleteConnection(int connId) {
    std::scoped_lock lk{g_state_mutex};
    auto it = g_connections.find(connId);
    if (it == g_connections.end()) {
        LOG_WARNING(Lib_Http, "sceHttpDeleteConnection: id={} not found", connId);
        return ORBIS_HTTP_ERROR_INVALID_ID;
    }
    g_connections.erase(it);
    LOG_DEBUG(Lib_Http, "sceHttpDeleteConnection: id={} freed", connId);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpDeleteRequest(int reqId) {
    std::scoped_lock lk{g_state_mutex};
    auto it = g_requests.find(reqId);
    if (it == g_requests.end()) {
        LOG_WARNING(Lib_Http, "sceHttpDeleteRequest: id={} not found", reqId);
        return ORBIS_HTTP_ERROR_INVALID_ID;
    }
    g_requests.erase(it);
    LOG_DEBUG(Lib_Http, "sceHttpDeleteRequest: id={} freed", reqId);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpDeleteTemplate(int tmplId) {
    std::scoped_lock lk{g_state_mutex};
    auto it = g_templates.find(tmplId);
    if (it == g_templates.end()) {
        LOG_WARNING(Lib_Http, "sceHttpDeleteTemplate: id={} not found", tmplId);
        return ORBIS_HTTP_ERROR_INVALID_ID;
    }
    g_templates.erase(it);
    LOG_DEBUG(Lib_Http, "sceHttpDeleteTemplate: id={} freed", tmplId);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpDestroyEpoll(int libhttpCtxId, void* eh) {
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetAcceptEncodingGZIPEnabled(int id, int* isEnable) {
    if (isEnable == nullptr) {
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lk{g_state_mutex};
    auto t = ResolveSettingsId(id);
    if (t.scope == SettingScope::Template) {
        *isEnable = t.tmpl->acceptEncodingGzip ? 1 : 0;
        return ORBIS_OK;
    }
    return ORBIS_HTTP_ERROR_INVALID_ID;
}

int PS4_SYSV_ABI sceHttpGetAllResponseHeaders(int reqId, char** header, u64* headerSize) {
    if (header == nullptr || headerSize == nullptr) {
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lk{g_state_mutex};
    auto it = g_requests.find(reqId);
    if (it == g_requests.end()) {
        return ORBIS_HTTP_ERROR_INVALID_ID;
    }
    HttpRequest& req = it->second;
    if (!req.sent) {
        return ORBIS_HTTP_ERROR_BEFORE_SEND;
    }
    *header = req.responseHeadersBlob.empty() ? nullptr
                                              : const_cast<char*>(req.responseHeadersBlob.data());
    *headerSize = req.responseHeadersBlob.size();
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetAuthEnabled(int id, int* isEnable) {
    if (isEnable == nullptr) {
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lk{g_state_mutex};
    auto t = ResolveSettingsId(id);
    switch (t.scope) {
    case SettingScope::Template:
        *isEnable = t.tmpl->authEnabled ? 1 : 0;
        return ORBIS_OK;
    case SettingScope::Connection:
        *isEnable = t.conn->authEnabled ? 1 : 0;
        return ORBIS_OK;
    default:
        return ORBIS_HTTP_ERROR_INVALID_ID;
    }
}

int PS4_SYSV_ABI sceHttpGetAutoRedirect(int id, int* isEnable) {
    if (isEnable == nullptr) {
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lk{g_state_mutex};
    auto t = ResolveSettingsId(id);
    switch (t.scope) {
    case SettingScope::Template:
        *isEnable = t.tmpl->autoRedirect ? 1 : 0;
        return ORBIS_OK;
    case SettingScope::Connection:
        *isEnable = t.conn->autoRedirect ? 1 : 0;
        return ORBIS_OK;
    default:
        return ORBIS_HTTP_ERROR_INVALID_ID;
    }
}

int PS4_SYSV_ABI sceHttpGetConnectionStat() {
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetCookie(int libhttpCtxId, const char* url, char* cookie, u64* required,
                                  u64 prepared, int isSecure) {
    if (url == nullptr || required == nullptr) {
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lk{g_state_mutex};
    std::string match;
    for (const auto& c : g_cookies) {
        if (std::strstr(url, c.url.c_str()) != nullptr) {
            if (!match.empty())
                match += "; ";
            match += c.raw;
        }
    }
    *required = match.size() + 1; // null terminator
    if (cookie != nullptr && prepared >= match.size() + 1) {
        std::memcpy(cookie, match.data(), match.size());
        cookie[match.size()] = '\0';
    }
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetCookieEnabled(int id, int* isEnable) {
    if (isEnable == nullptr) {
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lk{g_state_mutex};
    auto t = ResolveSettingsId(id);
    switch (t.scope) {
    case SettingScope::Template:
        *isEnable = t.tmpl->cookieEnabled ? 1 : 0;
        return ORBIS_OK;
    case SettingScope::Connection:
        *isEnable = t.conn->cookieEnabled ? 1 : 0;
        return ORBIS_OK;
    default:
        return ORBIS_HTTP_ERROR_INVALID_ID;
    }
}

int PS4_SYSV_ABI sceHttpGetCookieStats(int libhttpCtxId, void* stats) {
    if (stats != nullptr) {
        std::memset(stats, 0, 24);
    }
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetEpoll() {
    return ORBIS_OK;
}
int PS4_SYSV_ABI sceHttpGetEpollId() {
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetLastErrno(int reqId, int* errNum) {
    if (errNum)
        *errNum = 0;
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetMemoryPoolStats(int libhttpCtxId, void* stats) {
    // Three u64 fields (poolSize, maxInuseSize, currentInuseSize). All zero.
    if (stats != nullptr) {
        std::memset(stats, 0, 24);
    }
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetNonblock(int id, int* isEnable) {
    if (isEnable == nullptr) {
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lk{g_state_mutex};
    auto t = ResolveSettingsId(id);
    switch (t.scope) {
    case SettingScope::Template:
        *isEnable = t.tmpl->nonblock ? 1 : 0;
        return ORBIS_OK;
    case SettingScope::Connection:
        *isEnable = t.conn->nonblock ? 1 : 0;
        return ORBIS_OK;
    default:
        return ORBIS_HTTP_ERROR_INVALID_ID;
    }
}

int PS4_SYSV_ABI sceHttpGetRegisteredCtxIds(int* ids, u64* num) {
    // No multi-context model — the global httpInit gives one ctx.
    // Report empty list.
    if (num)
        *num = 0;
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetResponseContentLength(int reqId, int* result, u64* contentLength) {
    if (result == nullptr || contentLength == nullptr) {
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lk{g_state_mutex};
    auto it = g_requests.find(reqId);
    if (it == g_requests.end()) {
        return ORBIS_HTTP_ERROR_INVALID_ID;
    }
    HttpRequest& req = it->second;
    if (!req.sent) {
        return ORBIS_HTTP_ERROR_BEFORE_SEND;
    }
    *result = 0; // ORBIS_HTTP_CONTENTLEN_EXIST
    *contentLength = req.responseBody.size();
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetStatusCode(int reqId, int* statusCode) {
    if (statusCode == nullptr) {
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lk{g_state_mutex};
    auto it = g_requests.find(reqId);
    if (it == g_requests.end()) {
        LOG_WARNING(Lib_Http, "sceHttpGetStatusCode: req={} not found", reqId);
        return ORBIS_HTTP_ERROR_INVALID_ID;
    }
    HttpRequest& req = it->second;
    if (!req.sent) {
        return ORBIS_HTTP_ERROR_BEFORE_SEND;
    }
    *statusCode = req.statusCode;
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpInit(int libnetMemId, int libsslCtxId, u64 poolSize) {
    LOG_ERROR(Lib_Http, "(DUMMY) called libnetMemId = {} libsslCtxId = {} poolSize = {}",
              libnetMemId, libsslCtxId, poolSize);
    // return a value >1
    static int id = 0;
    return ++id;
}

int PS4_SYSV_ABI sceHttpParseResponseHeader(const char* header, u64 headerLen, const char* fieldStr,
                                            const char** fieldValue, u64* valueLen) {
    if (header == nullptr || fieldStr == nullptr || fieldValue == nullptr || valueLen == nullptr) {
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }
    const u64 fieldLen = std::strlen(fieldStr);
    if (fieldLen == 0) {
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }

    auto ci_eq = [](char a, char b) {
        if (a >= 'A' && a <= 'Z')
            a = char(a + 32);
        if (b >= 'A' && b <= 'Z')
            b = char(b + 32);
        return a == b;
    };

    const char* p = header;
    const char* end = header + headerLen;
    while (p < end) {
        const char* lineEnd = p;
        while (lineEnd < end && *lineEnd != '\n')
            ++lineEnd;
        if (static_cast<u64>(lineEnd - p) > fieldLen && p[fieldLen] == ':') {
            bool match = true;
            for (u64 i = 0; i < fieldLen; ++i) {
                if (!ci_eq(p[i], fieldStr[i])) {
                    match = false;
                    break;
                }
            }
            if (match) {
                const char* v = p + fieldLen + 1;
                while (v < lineEnd && (*v == ' ' || *v == '\t'))
                    ++v;
                const char* vEnd = lineEnd;
                while (vEnd > v && (vEnd[-1] == ' ' || vEnd[-1] == '\t' || vEnd[-1] == '\r'))
                    --vEnd;
                *fieldValue = v;
                *valueLen = static_cast<u64>(vEnd - v);
                return ORBIS_OK;
            }
        }
        p = lineEnd + 1;
    }

    *fieldValue = nullptr;
    *valueLen = 0;
    return ORBIS_HTTP_ERROR_PARSE_HTTP_NOT_FOUND;
}

int PS4_SYSV_ABI sceHttpParseStatusLine(const char* statusLine, u64 lineLen, int32_t* httpMajorVer,
                                        int32_t* httpMinorVer, int32_t* responseCode,
                                        const char** reasonPhrase, u64* phraseLen) {
    if (!statusLine) {
        LOG_ERROR(Lib_Http, "Invalid response");
        return ORBIS_HTTP_ERROR_PARSE_HTTP_INVALID_RESPONSE;
    }
    if (!httpMajorVer || !httpMinorVer || !responseCode || !reasonPhrase || !phraseLen) {
        LOG_ERROR(Lib_Http, "Invalid value");
        return ORBIS_HTTP_ERROR_PARSE_HTTP_INVALID_VALUE;
    }
    *httpMajorVer = 0;
    *httpMinorVer = 0;
    if (lineLen < 8) {
        LOG_ERROR(Lib_Http, "Linelen is smaller than 8");
        return ORBIS_HTTP_ERROR_PARSE_HTTP_INVALID_RESPONSE;
    }
    if (strncmp(statusLine, "HTTP/", 5) != 0) {
        LOG_ERROR(Lib_Http, "statusLine doesn't start with HTTP/");
        return ORBIS_HTTP_ERROR_PARSE_HTTP_INVALID_RESPONSE;
    }

    u64 index = 5;

    if (!isdigit(statusLine[index])) {
        LOG_ERROR(Lib_Http, "Invalid response");

        return ORBIS_HTTP_ERROR_PARSE_HTTP_INVALID_RESPONSE;
    }

    while (isdigit(statusLine[index])) {
        *httpMajorVer = *httpMajorVer * 10 + (statusLine[index] - '0');
        index++;
    }

    if (statusLine[index] != '.') {
        LOG_ERROR(Lib_Http, "Invalid response");
        return ORBIS_HTTP_ERROR_PARSE_HTTP_INVALID_RESPONSE;
    }
    index++;

    if (!isdigit(statusLine[index])) {
        LOG_ERROR(Lib_Http, "Invalid response");
        return ORBIS_HTTP_ERROR_PARSE_HTTP_INVALID_RESPONSE;
    }

    while (isdigit(statusLine[index])) {
        *httpMinorVer = *httpMinorVer * 10 + (statusLine[index] - '0');
        index++;
    }

    if (statusLine[index] != ' ') {
        LOG_ERROR(Lib_Http, "Invalid response");
        return ORBIS_HTTP_ERROR_PARSE_HTTP_INVALID_RESPONSE;
    }
    index++;

    // Validate and parse the 3-digit HTTP response code
    if (lineLen - index < 3 || !isdigit(statusLine[index]) || !isdigit(statusLine[index + 1]) ||
        !isdigit(statusLine[index + 2])) {
        LOG_ERROR(Lib_Http, "Invalid response");
        return ORBIS_HTTP_ERROR_PARSE_HTTP_INVALID_RESPONSE;
    }

    *responseCode = (statusLine[index] - '0') * 100 + (statusLine[index + 1] - '0') * 10 +
                    (statusLine[index + 2] - '0');
    index += 3;

    if (statusLine[index] != ' ') {
        LOG_ERROR(Lib_Http, "Invalid response");
        return ORBIS_HTTP_ERROR_PARSE_HTTP_INVALID_RESPONSE;
    }
    index++;

    // Set the reason phrase start position
    *reasonPhrase = &statusLine[index];
    u64 phraseStart = index;

    while (index < lineLen && statusLine[index] != '\n') {
        index++;
    }

    // Determine the length of the reason phrase, excluding trailing \r if present
    if (index == phraseStart) {
        *phraseLen = 0;
    } else {
        *phraseLen =
            (statusLine[index - 1] == '\r') ? (index - phraseStart - 1) : (index - phraseStart);
    }

    // Return the number of bytes processed
    return index + 1;
}

int PS4_SYSV_ABI sceHttpReadData(s32 reqId, void* data, u64 size) {
    if (data == nullptr || size == 0) {
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lk{g_state_mutex};
    auto it = g_requests.find(reqId);
    if (it == g_requests.end()) {
        LOG_WARNING(Lib_Http, "sceHttpReadData: req={} not found", reqId);
        return ORBIS_HTTP_ERROR_INVALID_ID;
    }
    HttpRequest& req = it->second;
    if (!req.sent) {
        // Real SDK returns BEFORE_SEND if you try to read without sending.
        return ORBIS_HTTP_ERROR_BEFORE_SEND;
    }
    const size_t available = req.responseBody.size() - req.readOffset;
    const size_t toCopy = std::min<size_t>(static_cast<size_t>(size), available);
    if (toCopy > 0) {
        std::memcpy(data, req.responseBody.data() + req.readOffset, toCopy);
        req.readOffset += toCopy;
    }
    // Returns bytes read; 0 signals end-of-stream to callers.
    return static_cast<int>(toCopy);
}

int PS4_SYSV_ABI sceHttpRedirectCacheFlush(int libhttpCtxId) {
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpRemoveRequestHeader(int reqId, const char* name) {
    if (name == nullptr) {
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lk{g_state_mutex};
    auto it = g_requests.find(reqId);
    if (it == g_requests.end()) {
        return ORBIS_HTTP_ERROR_INVALID_ID;
    }
    HttpRequest& req = it->second;
    for (auto h = req.requestHeaders.begin(); h != req.requestHeaders.end();) {
        if (h->first == name) {
            h = req.requestHeaders.erase(h);
        } else {
            ++h;
        }
    }
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpRequestGetAllHeaders(int reqId, char** header, u64* headerSize) {
    if (header == nullptr || headerSize == nullptr) {
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lk{g_state_mutex};
    auto it = g_requests.find(reqId);
    if (it == g_requests.end()) {
        return ORBIS_HTTP_ERROR_INVALID_ID;
    }
    HttpRequest& req = it->second;
    std::string blob;
    for (const auto& [k, v] : req.requestHeaders) {
        blob += k;
        blob += ": ";
        blob += v;
        blob += "\r\n";
    }
    req.responseHeadersBlob = std::move(blob);
    *header = req.responseHeadersBlob.empty() ? nullptr
                                              : const_cast<char*>(req.responseHeadersBlob.data());
    *headerSize = req.responseHeadersBlob.size();
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpsDisableOption(int id, uint32_t sslFlags) {
    std::scoped_lock lk{g_state_mutex};
    auto t = ResolveSettingsId(id);
    if (t.scope == SettingScope::Template) {
        t.tmpl->sslFlags &= ~sslFlags;
        return ORBIS_OK;
    }
    return ORBIS_HTTP_ERROR_INVALID_ID;
}

int PS4_SYSV_ABI sceHttpsDisableOptionPrivate(int id, uint32_t sslFlags) {
    return sceHttpsDisableOption(id, sslFlags);
}

int PS4_SYSV_ABI sceHttpsEnableOption(int id, uint32_t sslFlags) {
    std::scoped_lock lk{g_state_mutex};
    auto t = ResolveSettingsId(id);
    if (t.scope == SettingScope::Template) {
        t.tmpl->sslFlags |= sslFlags;
        return ORBIS_OK;
    }
    return ORBIS_HTTP_ERROR_INVALID_ID;
}

int PS4_SYSV_ABI sceHttpsEnableOptionPrivate(int id, uint32_t sslFlags) {
    return sceHttpsEnableOption(id, sslFlags);
}

int PS4_SYSV_ABI sceHttpSendRequest(int reqId, const void* postData, u64 size) {
    HttpConnection* connPtr = nullptr;
    std::string method;
    std::string path;
    std::vector<std::pair<std::string, std::string>> headers;
    {
        std::scoped_lock lk{g_state_mutex};
        auto reqIt = g_requests.find(reqId);
        if (reqIt == g_requests.end()) {
            LOG_ERROR(Lib_Http, "sceHttpSendRequest: req={} not found", reqId);
            return ORBIS_HTTP_ERROR_INVALID_ID;
        }
        HttpRequest& req = reqIt->second;
        if (req.sent) {
            LOG_WARNING(Lib_Http, "sceHttpSendRequest: req={} already sent", reqId);
            return ORBIS_HTTP_ERROR_AFTER_SEND;
        }
        auto connIt = g_connections.find(req.connectionId);
        if (connIt == g_connections.end()) {
            LOG_ERROR(Lib_Http, "sceHttpSendRequest: req={} conn={} not found", reqId,
                      req.connectionId);
            return ORBIS_HTTP_ERROR_INVALID_ID;
        }
        connPtr = &connIt->second;
        const char* methodStr = MethodName(req.method);
        if (methodStr == nullptr) {
            LOG_WARNING(Lib_Http, "sceHttpSendRequest: req={} unknown method enum {}", reqId,
                        req.method);
            return ORBIS_HTTP_ERROR_UNKNOWN_METHOD;
        }
        method = methodStr;
        path = req.path;
        headers = req.requestHeaders;
        req.sent = true;
    }

    // Build the headers map cpp-httplib expects.
    httplib::Headers httplibHeaders;
    for (const auto& [name, value] : headers) {
        httplibHeaders.emplace(name, value);
    }

    httplib::Client* cli = GetOrCreateClient(*connPtr);
    httplib::Result res;

    const std::string body =
        (postData != nullptr && size > 0)
            ? std::string(static_cast<const char*>(postData), static_cast<size_t>(size))
            : std::string();

    std::string contentType;
    for (const auto& [name, value] : headers) {
        if (name == "Content-Type" || name == "content-type") {
            contentType = value;
            break;
        }
    }

    if (method == "GET") {
        res = cli->Get(path, httplibHeaders);
    } else if (method == "DELETE") {
        res = cli->Delete(path, httplibHeaders);
    } else if (method == "HEAD") {
        res = cli->Head(path, httplibHeaders);
    } else if (method == "OPTIONS") {
        res = cli->Options(path, httplibHeaders);
    } else if (method == "POST") {
        res = cli->Post(path, httplibHeaders, body, contentType);
    } else if (method == "PUT") {
        res = cli->Put(path, httplibHeaders, body, contentType);
    } else if (method == "PATCH") {
        res = cli->Patch(path, httplibHeaders, body, contentType);
    } else {
        LOG_WARNING(Lib_Http, "sceHttpSendRequest: req={} unsupported method '{}'", reqId, method);
        std::scoped_lock lk{g_state_mutex};
        auto reqIt = g_requests.find(reqId);
        if (reqIt != g_requests.end()) {
            reqIt->second.sent = false;
        }
        return ORBIS_HTTP_ERROR_UNKNOWN_METHOD;
    }

    if (!res) {
        const auto err = res.error();
        LOG_ERROR(Lib_Http, "sceHttpSendRequest: req={} {} {} network failure: {}", reqId, method,
                  path, httplib::to_string(err));
        switch (err) {
        case httplib::Error::Connection:
        case httplib::Error::ConnectionTimeout:
            return ORBIS_HTTP_ERROR_NETWORK;
        case httplib::Error::Read:
        case httplib::Error::Write:
            return ORBIS_HTTP_ERROR_BROKEN;
        case httplib::Error::SSLConnection:
        case httplib::Error::SSLLoadingCerts:
        case httplib::Error::SSLServerVerification:
            return ORBIS_HTTP_ERROR_SSL;
        default:
            return ORBIS_HTTP_ERROR_UNKNOWN;
        }
    }

    // Stash the response on the request record so subsequent
    // sceHttpReadData / sceHttpGetStatusCode / sceHttpGetAllResponseHeaders
    // calls can read it.
    {
        std::scoped_lock lk{g_state_mutex};
        auto reqIt = g_requests.find(reqId);
        if (reqIt == g_requests.end()) {
            // Request was deleted while we were on the wire. Drop the
            // response cause nothing left to write into.
            LOG_WARNING(Lib_Http, "sceHttpSendRequest: req={} deleted during send", reqId);
            return ORBIS_HTTP_ERROR_INVALID_ID;
        }
        HttpRequest& req = reqIt->second;
        req.statusCode = res->status;
        req.responseBody = std::move(res->body);
        req.readOffset = 0;
        req.responseHeadersBlob = SerializeHeadersBlob(res->headers);
    }

    size_t bodySize = 0;
    {
        std::scoped_lock lk{g_state_mutex};
        auto it = g_requests.find(reqId);
        if (it != g_requests.end()) {
            bodySize = it->second.responseBody.size();
        }
    }
    LOG_INFO(Lib_Http, "sceHttpSendRequest: req={} {} {} -> HTTP {} ({} body bytes)", reqId, method,
             path, res->status, bodySize);
    return ORBIS_OK;
}

// Helper: simple boolean setter that toggles a per-template-or-connection
// flag, dispatching via ResolveSettingsId. Used by all the SetXxxEnabled
// functions below.
static int SetBoolFlag(int id, int isEnable, bool HttpTemplate::* tmplField,
                       bool HttpConnection::* connField) {
    std::scoped_lock lk{g_state_mutex};
    auto t = ResolveSettingsId(id);
    switch (t.scope) {
    case SettingScope::Template:
        if (tmplField)
            t.tmpl->*tmplField = (isEnable != 0);
        return ORBIS_OK;
    case SettingScope::Connection:
        if (connField)
            t.conn->*connField = (isEnable != 0);
        return ORBIS_OK;
    default:
        return ORBIS_HTTP_ERROR_INVALID_ID;
    }
}

int PS4_SYSV_ABI sceHttpSetAcceptEncodingGZIPEnabled(int id, int isEnable) {
    return SetBoolFlag(id, isEnable, &HttpTemplate::acceptEncodingGzip, nullptr);
}

int PS4_SYSV_ABI sceHttpSetAuthEnabled(int id, int isEnable) {
    return SetBoolFlag(id, isEnable, &HttpTemplate::authEnabled, &HttpConnection::authEnabled);
}

int PS4_SYSV_ABI sceHttpSetAuthInfoCallback(int id, void* cbfunc, void* userArg) {
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetAutoRedirect(int id, int isEnable) {
    return SetBoolFlag(id, isEnable, &HttpTemplate::autoRedirect, &HttpConnection::autoRedirect);
}

int PS4_SYSV_ABI sceHttpSetChunkedTransferEnabled(int id, int isEnable) {
    return SetBoolFlag(id, isEnable, &HttpTemplate::chunkedTransfer, nullptr);
}

int PS4_SYSV_ABI sceHttpSetConnectTimeOut(int id, uint32_t usec) {
    std::scoped_lock lk{g_state_mutex};
    auto t = ResolveSettingsId(id);
    switch (t.scope) {
    case SettingScope::Template:
        t.tmpl->connectTimeoutUs = usec;
        return ORBIS_OK;
    case SettingScope::Connection:
        t.conn->connectTimeoutUs = usec;
        if (t.conn->client) {
            t.conn->client->set_connection_timeout(std::chrono::microseconds(usec));
        }
        return ORBIS_OK;
    default:
        return ORBIS_HTTP_ERROR_INVALID_ID;
    }
}

int PS4_SYSV_ABI sceHttpSetCookieEnabled(int id, int isEnable) {
    return SetBoolFlag(id, isEnable, &HttpTemplate::cookieEnabled, &HttpConnection::cookieEnabled);
}

int PS4_SYSV_ABI sceHttpSetCookieMaxNum(int libhttpCtxId, uint32_t num) {
    return ORBIS_OK;
}
int PS4_SYSV_ABI sceHttpSetCookieMaxNumPerDomain(int libhttpCtxId, uint32_t num) {
    return ORBIS_OK;
}
int PS4_SYSV_ABI sceHttpSetCookieMaxSize(int libhttpCtxId, uint32_t size) {
    return ORBIS_OK;
}
int PS4_SYSV_ABI sceHttpSetCookieRecvCallback(int id, void* cbfunc, void* userArg) {
    return ORBIS_OK;
}
int PS4_SYSV_ABI sceHttpSetCookieSendCallback(int id, void* cbfunc, void* userArg) {
    return ORBIS_OK;
}
int PS4_SYSV_ABI sceHttpSetCookieTotalMaxSize(int libhttpCtxId, uint32_t size) {
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetDefaultAcceptEncodingGZIPEnabled(int libhttpCtxId, int isEnable) {
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetDelayBuildRequestEnabled(int id, int isEnable) {
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetEpoll(int id, void* eh, void* hints) {
    return ORBIS_OK;
}
int PS4_SYSV_ABI sceHttpSetEpollId(int id, void* eh) {
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetHttp09Enabled(int id, int isEnable) {
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetInflateGZIPEnabled(int id, int isEnable) {
    return SetBoolFlag(id, isEnable, &HttpTemplate::inflateGzip, nullptr);
}

int PS4_SYSV_ABI sceHttpSetNonblock(int id, int isEnable) {
    return SetBoolFlag(id, isEnable, &HttpTemplate::nonblock, &HttpConnection::nonblock);
}

int PS4_SYSV_ABI sceHttpSetPolicyOption(int libhttpCtxId, int policy) {
    // Connection-policy hints (use IPv6, prefer keepalive, etc.). No
    // direct httplib equivalent for most; ignore.
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetPriorityOption(int libhttpCtxId, int priority) {
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetProxy(int id, int mode, int version, const char* host, u16 port) {
    std::scoped_lock lk{g_state_mutex};
    auto t = ResolveSettingsId(id);
    if (t.scope == SettingScope::Template) {
        t.tmpl->proxyMode = mode;
        t.tmpl->proxyVersion = version;
        t.tmpl->proxyHost = host ? std::string(host) : std::string{};
        t.tmpl->proxyPort = port;
        return ORBIS_OK;
    }
    return ORBIS_HTTP_ERROR_INVALID_ID;
}

int PS4_SYSV_ABI sceHttpSetRecvBlockSize(int id, uint32_t blockSize) {
    std::scoped_lock lk{g_state_mutex};
    auto t = ResolveSettingsId(id);
    if (t.scope == SettingScope::Template) {
        t.tmpl->recvBlockSize = static_cast<int>(blockSize);
        return ORBIS_OK;
    }
    return ORBIS_HTTP_ERROR_INVALID_ID;
}

int PS4_SYSV_ABI sceHttpSetRecvTimeOut(int id, uint32_t usec) {
    std::scoped_lock lk{g_state_mutex};
    auto t = ResolveSettingsId(id);
    switch (t.scope) {
    case SettingScope::Template:
        t.tmpl->recvTimeoutUs = usec;
        return ORBIS_OK;
    case SettingScope::Connection:
        t.conn->recvTimeoutUs = usec;
        if (t.conn->client) {
            t.conn->client->set_read_timeout(std::chrono::microseconds(usec));
        }
        return ORBIS_OK;
    default:
        return ORBIS_HTTP_ERROR_INVALID_ID;
    }
}

int PS4_SYSV_ABI sceHttpSetRedirectCallback(int id, void* cbfunc, void* userArg) {
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetRequestContentLength(int id, u64 contentLength) {
    std::scoped_lock lk{g_state_mutex};
    auto it = g_requests.find(id);
    if (it == g_requests.end()) {
        return ORBIS_HTTP_ERROR_INVALID_ID;
    }
    it->second.contentLength = contentLength;
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetRequestStatusCallback(int id, void* cbfunc, void* userArg) {
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetResolveRetry(int id, int retry) {
    std::scoped_lock lk{g_state_mutex};
    auto t = ResolveSettingsId(id);
    if (t.scope == SettingScope::Template) {
        t.tmpl->resolveRetry = static_cast<uint32_t>(retry);
        return ORBIS_OK;
    }
    return ORBIS_HTTP_ERROR_INVALID_ID;
}

int PS4_SYSV_ABI sceHttpSetResolveTimeOut(int id, uint32_t usec) {
    std::scoped_lock lk{g_state_mutex};
    auto t = ResolveSettingsId(id);
    switch (t.scope) {
    case SettingScope::Template:
        t.tmpl->resolveTimeoutUs = usec;
        return ORBIS_OK;
    case SettingScope::Connection:
        t.conn->resolveTimeoutUs = usec;
        return ORBIS_OK;
    default:
        return ORBIS_HTTP_ERROR_INVALID_ID;
    }
}

int PS4_SYSV_ABI sceHttpSetResponseHeaderMaxSize(int id, u64 headerSize) {
    std::scoped_lock lk{g_state_mutex};
    auto t = ResolveSettingsId(id);
    if (t.scope == SettingScope::Template) {
        t.tmpl->responseHeaderMaxSize = static_cast<int>(headerSize);
        return ORBIS_OK;
    }
    return ORBIS_HTTP_ERROR_INVALID_ID;
}

int PS4_SYSV_ABI sceHttpSetSendTimeOut(int id, uint32_t usec) {
    std::scoped_lock lk{g_state_mutex};
    auto t = ResolveSettingsId(id);
    switch (t.scope) {
    case SettingScope::Template:
        t.tmpl->sendTimeoutUs = usec;
        return ORBIS_OK;
    case SettingScope::Connection:
        t.conn->sendTimeoutUs = usec;
        if (t.conn->client) {
            t.conn->client->set_write_timeout(std::chrono::microseconds(usec));
        }
        return ORBIS_OK;
    default:
        return ORBIS_HTTP_ERROR_INVALID_ID;
    }
}

int PS4_SYSV_ABI sceHttpSetSocketCreationCallback(int libhttpCtxId, void* cbfunc, void* userArg) {
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpsFreeCaList(int libhttpCtxId, void* caList) {
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpsGetCaList(int httpCtxId, OrbisHttpsCaList* list) {
    LOG_ERROR(Lib_Http, "(DUMMY) called, httpCtxId = {}", httpCtxId);
    list->certsNum = 0;
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpsGetSslError(int id, int* errNum, uint32_t* detail) {
    if (errNum)
        *errNum = 0;
    if (detail)
        *detail = 0;
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpsLoadCert(int libhttpCtxId, int caCertNum, const void** caList,
                                  const void* cert, const void* privKey) {
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpsSetMinSslVersion(int id, int version) {
    std::scoped_lock lk{g_state_mutex};
    auto t = ResolveSettingsId(id);
    if (t.scope == SettingScope::Template) {
        t.tmpl->sslMinVersion = version;
        return ORBIS_OK;
    }
    return ORBIS_HTTP_ERROR_INVALID_ID;
}

int PS4_SYSV_ABI sceHttpsSetSslCallback(int id, void* cbfunc, void* userArg) {
    // Game-side cert verification callback. No HTTPS path to call it on.
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpsSetSslVersion(int id, int version) {
    return sceHttpsSetMinSslVersion(id, version);
}

int PS4_SYSV_ABI sceHttpsUnloadCert(int libhttpCtxId) {
    // Symmetric with sceHttpsLoadCert. No state to free.
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpTerm(int libhttpCtxId) {
    std::scoped_lock lk{g_state_mutex};
    g_requests.clear();
    g_connections.clear();
    g_templates.clear();
    g_cookies.clear();
    g_next_template_id = 1;
    g_next_connection_id = 1;
    g_next_request_id = 1;
    g_isHttpInitialized = false;
    LOG_INFO(Lib_Http, "sceHttpTerm: state cleared");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpTryGetNonblock(int id, int* isEnable) {
    return sceHttpGetNonblock(id, isEnable);
}

int PS4_SYSV_ABI sceHttpTrySetNonblock(int id, int isEnable) {
    return sceHttpSetNonblock(id, isEnable);
}

int PS4_SYSV_ABI sceHttpUnsetEpoll(int id) {
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpUriBuild(char* out, u64* require, u64 prepare,
                                 const OrbisHttpUriElement* srcElement, u32 option) {
    if (require == nullptr || srcElement == nullptr) {
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }
    const u32 OPT_SCHEME = 0x01, OPT_HOSTNAME = 0x02, OPT_PORT = 0x04, OPT_PATH = 0x08,
              OPT_USERNAME = 0x10, OPT_PASSWORD = 0x20, OPT_QUERY = 0x40, OPT_FRAGMENT = 0x80;

    std::string built;
    if ((option & OPT_SCHEME) && srcElement->scheme) {
        built.append(srcElement->scheme);
        built.append("://");
    }
    if ((option & OPT_USERNAME) && srcElement->username) {
        built.append(srcElement->username);
        if ((option & OPT_PASSWORD) && srcElement->password) {
            built.append(":");
            built.append(srcElement->password);
        }
        built.append("@");
    }
    if ((option & OPT_HOSTNAME) && srcElement->hostname) {
        built.append(srcElement->hostname);
    }
    if ((option & OPT_PORT) && srcElement->port != 0) {
        built.append(":");
        built.append(std::to_string(srcElement->port));
    }
    if ((option & OPT_PATH) && srcElement->path) {
        built.append(srcElement->path);
    }
    if ((option & OPT_QUERY) && srcElement->query) {
        built.append(srcElement->query);
    }
    if ((option & OPT_FRAGMENT) && srcElement->fragment) {
        built.append(srcElement->fragment);
    }
    *require = built.size() + 1;
    if (out != nullptr) {
        if (prepare < built.size() + 1) {
            return ORBIS_HTTP_ERROR_OUT_OF_SIZE;
        }
        std::memcpy(out, built.data(), built.size());
        out[built.size()] = '\0';
    }
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpUriCopy(OrbisHttpUriElement* dest, const OrbisHttpUriElement* src,
                                void* pool, u64* require, u64 prepare) {
    if (src == nullptr || require == nullptr) {
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }
    auto strLen = [](const char* s) -> u64 { return s ? (std::strlen(s) + 1) : 0; };
    const u64 needed = strLen(src->scheme) + strLen(src->username) + strLen(src->password) +
                       strLen(src->hostname) + strLen(src->path) + strLen(src->query) +
                       strLen(src->fragment);
    *require = needed;
    if (dest == nullptr || pool == nullptr || prepare < needed) {
        return ORBIS_OK;
    }

    char* p = static_cast<char*>(pool);
    auto put = [&](const char* s) -> char* {
        if (!s)
            return nullptr;
        const u64 n = std::strlen(s) + 1;
        std::memcpy(p, s, n);
        char* out = p;
        p += n;
        return out;
    };
    *dest = *src;
    dest->scheme = put(src->scheme);
    dest->username = put(src->username);
    dest->password = put(src->password);
    dest->hostname = put(src->hostname);
    dest->path = put(src->path);
    dest->query = put(src->query);
    dest->fragment = put(src->fragment);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpUriEscape(char* out, u64* require, u64 prepare, const char* in) {
    LOG_TRACE(Lib_Http, "called");

    if (!in) {
        LOG_ERROR(Lib_Http, "Invalid input string");
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }

    auto IsUnreserved = [](unsigned char c) -> bool {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
               c == '-' || c == '_' || c == '.' || c == '~';
    };

    u64 needed = 0;
    const char* src = in;
    while (*src) {
        unsigned char c = static_cast<unsigned char>(*src);
        if (IsUnreserved(c)) {
            needed++;
        } else {
            needed += 3; // %XX format
        }
        src++;
    }
    needed++; // null terminator

    if (require) {
        *require = needed;
    }

    if (!out) {
        return ORBIS_OK;
    }

    if (prepare < needed) {
        LOG_ERROR(Lib_Http, "Buffer too small: need {} but only {} available", needed, prepare);
        return ORBIS_HTTP_ERROR_OUT_OF_MEMORY;
    }

    static const char hex_chars[] = "0123456789ABCDEF";
    src = in;
    char* dst = out;
    while (*src) {
        unsigned char c = static_cast<unsigned char>(*src);
        if (IsUnreserved(c)) {
            *dst++ = *src;
        } else {
            *dst++ = '%';
            *dst++ = hex_chars[(c >> 4) & 0x0F];
            *dst++ = hex_chars[c & 0x0F];
        }
        src++;
    }
    *dst = '\0';

    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpUriMerge(char* mergedUrl, char* url, char* relativeUri, u64* require,
                                 u64 prepare, u32 option) {
    u64 requiredLength;
    int returnValue;
    u64 baseUrlLength;
    u64 relativeUriLength;
    u64 totalLength;
    u64 combinedLength;
    int parseResult;
    u64 localSizeRelativeUri;
    u64 localSizeBaseUrl;
    OrbisHttpUriElement parsedUriElement;

    if (option != 0 || url == NULL || relativeUri == NULL) {
        LOG_ERROR(Lib_Http, "Invalid value");
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }

    returnValue = sceHttpUriParse(NULL, url, NULL, &localSizeBaseUrl, 0);
    if (returnValue < 0) {
        LOG_ERROR(Lib_Http, "returning {:#x}", returnValue);
        return returnValue;
    }

    returnValue = sceHttpUriParse(NULL, relativeUri, NULL, &localSizeRelativeUri, 0);
    if (returnValue < 0) {
        LOG_ERROR(Lib_Http, "returning {:#x}", returnValue);
        return returnValue;
    }

    baseUrlLength = strnlen(url, 0x3fff);
    relativeUriLength = strnlen(relativeUri, 0x3fff);
    requiredLength = localSizeBaseUrl + 2 + (relativeUriLength + baseUrlLength) * 2;

    if (require) {
        *require = requiredLength;
    }

    if (mergedUrl == NULL) {
        return ORBIS_OK;
    }

    if (prepare < requiredLength) {
        LOG_ERROR(Lib_Http, "Error Out of memory");
        return ORBIS_HTTP_ERROR_OUT_OF_MEMORY;
    }

    totalLength = strnlen(url, 0x3fff);
    baseUrlLength = strnlen(relativeUri, 0x3fff);
    combinedLength = totalLength + 1 + baseUrlLength;
    relativeUriLength = prepare - combinedLength;

    returnValue =
        sceHttpUriParse(&parsedUriElement, relativeUri, mergedUrl + totalLength + baseUrlLength + 1,
                        &localSizeRelativeUri, relativeUriLength);
    if (returnValue < 0) {
        LOG_ERROR(Lib_Http, "returning {:#x}", returnValue);
        return returnValue;
    }
    if (parsedUriElement.scheme == NULL) {
        strncpy(mergedUrl, relativeUri, requiredLength);
        if (require) {
            *require = strnlen(relativeUri, 0x3fff) + 1;
        }
        return ORBIS_OK;
    }

    returnValue =
        sceHttpUriParse(&parsedUriElement, url, mergedUrl + totalLength + baseUrlLength + 1,
                        &localSizeBaseUrl, relativeUriLength);
    if (returnValue < 0) {
        LOG_ERROR(Lib_Http, "returning {:#x}", returnValue);
        return returnValue;
    }

    combinedLength += localSizeBaseUrl;
    strncpy(mergedUrl + combinedLength, parsedUriElement.path, prepare - combinedLength);
    NormalizeAndAppendPath(mergedUrl + combinedLength, relativeUri);

    returnValue = sceHttpUriBuild(mergedUrl, 0, ~(baseUrlLength + totalLength) + prepare,
                                  &parsedUriElement, 0x3f);
    if (returnValue >= 0) {
        return ORBIS_OK;
    } else {
        LOG_ERROR(Lib_Http, "returning {:#x}", returnValue);
        return returnValue;
    }
}

int PS4_SYSV_ABI sceHttpUriParse(OrbisHttpUriElement* out, const char* srcUri, void* pool,
                                 u64* require, u64 prepare) {
    LOG_INFO(Lib_Http, "srcUri = {}", std::string(srcUri));
    if (!srcUri) {
        LOG_ERROR(Lib_Http, "invalid url");
        return ORBIS_HTTP_ERROR_INVALID_URL;
    }
    if (!out && !pool && !require) {
        LOG_ERROR(Lib_Http, "invalid values");
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }

    if (out && pool) {
        memset(out, 0, sizeof(OrbisHttpUriElement));
        out->scheme = (char*)pool;
    }

    // Track the total required buffer size
    u64 requiredSize = 0;

    // Parse the scheme (e.g., "http:", "https:", "file:")
    u64 schemeLength = 0;
    while (srcUri[schemeLength] && srcUri[schemeLength] != ':') {
        if (!isalnum(srcUri[schemeLength])) {
            LOG_ERROR(Lib_Http, "invalid url");
            return ORBIS_HTTP_ERROR_INVALID_URL;
        }
        schemeLength++;
    }

    if (pool && prepare < schemeLength + 1) {
        LOG_ERROR(Lib_Http, "out of memory");
        return ORBIS_HTTP_ERROR_OUT_OF_MEMORY;
    }

    if (out && pool) {
        memcpy(out->scheme, srcUri, schemeLength);
        out->scheme[schemeLength] = '\0';
    }

    requiredSize += schemeLength + 1;

    // Move past the scheme and ':' character
    u64 offset = schemeLength + 1;

    // Check if "//" appears after the scheme
    if (strncmp(srcUri + offset, "//", 2) == 0) {
        // "//" is present
        if (out) {
            out->opaque = false;
        }
        offset += 2; // Move past "//"
    } else {
        // "//" is not present
        if (out) {
            out->opaque = true;
        }
    }

    // Handle "file" scheme
    if (strncmp(srcUri, "file", 4) == 0) {
        // File URIs typically start with "file://"
        if (out && !out->opaque) {
            // Skip additional slashes (e.g., "////")
            while (srcUri[offset] == '/') {
                offset++;
            }

            // Parse the path (everything after the slashes)
            char* pathStart = (char*)srcUri + offset;
            u64 pathLength = 0;
            while (pathStart[pathLength] && pathStart[pathLength] != '?' &&
                   pathStart[pathLength] != '#') {
                pathLength++;
            }

            if (pathLength > 0) {
                // Prepend '/' to the path
                requiredSize += pathLength + 2; // Include '/' and null terminator

                if (pool && prepare < requiredSize) {
                    LOG_ERROR(Lib_Http, "out of memory, provided size: {}, required size: {}",
                              prepare, requiredSize);
                    return ORBIS_HTTP_ERROR_OUT_OF_MEMORY;
                }

                if (out && pool) {
                    out->path = (char*)pool + (requiredSize - pathLength - 2);
                    out->username = (char*)pool + (requiredSize - pathLength - 3);
                    out->password = (char*)pool + (requiredSize - pathLength - 3);
                    out->hostname = (char*)pool + (requiredSize - pathLength - 3);
                    out->query = (char*)pool + (requiredSize - pathLength - 3);
                    out->fragment = (char*)pool + (requiredSize - pathLength - 3);
                    out->username[0] = '\0';
                    out->path[0] = '/'; // Add leading '/'
                    memcpy(out->path + 1, pathStart, pathLength);
                    out->path[pathLength + 1] = '\0';
                }
            } else {
                // Path already starts with '/'
                requiredSize += pathLength + 1;

                if (pool && prepare < requiredSize) {
                    LOG_ERROR(Lib_Http, "out of memory");
                    return ORBIS_HTTP_ERROR_OUT_OF_MEMORY;
                }

                if (out && pool) {
                    memcpy((char*)pool + (requiredSize - pathLength - 1), pathStart, pathLength);
                    out->path = (char*)pool + (requiredSize - pathLength - 1);
                    out->path[pathLength] = '\0';
                }
            }

            // Move past the path
            offset += pathLength;
        } else {
            // Parse the path (everything after the slashes)
            char* pathStart = (char*)srcUri + offset;
            u64 pathLength = 0;
            while (pathStart[pathLength] && pathStart[pathLength] != '?' &&
                   pathStart[pathLength] != '#') {
                pathLength++;
            }

            if (pathLength > 0) {
                requiredSize += pathLength + 3; // Add '/' and null terminator, and the dummy
                                                // null character for the other fields
            }
        }
    }

    // Handle non-file schemes (e.g., "http", "https")
    else {
        // Parse the host and port
        char* hostStart = (char*)srcUri + offset;
        while (*hostStart == '/') {
            hostStart++;
        }

        u64 hostLength = 0;
        while (hostStart[hostLength] && hostStart[hostLength] != '/' &&
               hostStart[hostLength] != '?' && hostStart[hostLength] != ':') {
            hostLength++;
        }

        requiredSize += hostLength + 1;

        if (pool && prepare < requiredSize) {
            LOG_ERROR(Lib_Http, "out of memory");
            return ORBIS_HTTP_ERROR_OUT_OF_MEMORY;
        }

        if (out && pool) {
            memcpy((char*)pool + (requiredSize - hostLength - 1), hostStart, hostLength);
            out->hostname = (char*)pool + (requiredSize - hostLength - 1);
            out->hostname[hostLength] = '\0';
        }

        // Move past the host
        offset += hostLength;

        // Parse the port (if present)
        if (hostStart[hostLength] == ':') {
            char* portStart = hostStart + hostLength + 1;
            u64 portLength = 0;
            while (portStart[portLength] && isdigit(portStart[portLength])) {
                portLength++;
            }

            requiredSize += portLength + 1;

            if (pool && prepare < requiredSize) {
                LOG_ERROR(Lib_Http, "out of memory");
                return ORBIS_HTTP_ERROR_OUT_OF_MEMORY;
            }

            // Convert the port string to a uint16_t
            char portStr[6]; // Max length for a port number (65535)
            if (portLength > 5) {
                LOG_ERROR(Lib_Http, "invalid url");
                return ORBIS_HTTP_ERROR_INVALID_URL;
            }
            memcpy(portStr, portStart, portLength);
            portStr[portLength] = '\0';

            uint16_t port = (uint16_t)atoi(portStr);
            if (port == 0 && portStr[0] != '0') {
                LOG_ERROR(Lib_Http, "invalid url");
                return ORBIS_HTTP_ERROR_INVALID_URL;
            }

            // Set the port in the output structure
            if (out) {
                out->port = port;
            }

            // Move past the port
            offset += portLength + 1;
        }
    }

    // Parse the path (if present)
    if (srcUri[offset] == '/') {
        char* pathStart = (char*)srcUri + offset;
        u64 pathLength = 0;
        while (pathStart[pathLength] && pathStart[pathLength] != '?' &&
               pathStart[pathLength] != '#') {
            pathLength++;
        }

        requiredSize += pathLength + 1;

        if (pool && prepare < requiredSize) {
            LOG_ERROR(Lib_Http, "out of memory");
            return ORBIS_HTTP_ERROR_OUT_OF_MEMORY;
        }

        if (out && pool) {
            memcpy((char*)pool + (requiredSize - pathLength - 1), pathStart, pathLength);
            out->path = (char*)pool + (requiredSize - pathLength - 1);
            out->path[pathLength] = '\0';
        }

        // Move past the path
        offset += pathLength;
    }

    // Parse the query (if present)
    if (srcUri[offset] == '?') {
        char* queryStart = (char*)srcUri + offset + 1;
        u64 queryLength = 0;
        while (queryStart[queryLength] && queryStart[queryLength] != '#') {
            queryLength++;
        }

        requiredSize += queryLength + 1;

        if (pool && prepare < requiredSize) {
            LOG_ERROR(Lib_Http, "out of memory");
            return ORBIS_HTTP_ERROR_OUT_OF_MEMORY;
        }

        if (out && pool) {
            memcpy((char*)pool + (requiredSize - queryLength - 1), queryStart, queryLength);
            out->query = (char*)pool + (requiredSize - queryLength - 1);
            out->query[queryLength] = '\0';
        }

        // Move past the query
        offset += queryLength + 1;
    }

    // Parse the fragment (if present)
    if (srcUri[offset] == '#') {
        char* fragmentStart = (char*)srcUri + offset + 1;
        u64 fragmentLength = 0;
        while (fragmentStart[fragmentLength]) {
            fragmentLength++;
        }

        requiredSize += fragmentLength + 1;

        if (pool && prepare < requiredSize) {
            LOG_ERROR(Lib_Http, "out of memory");
            return ORBIS_HTTP_ERROR_OUT_OF_MEMORY;
        }

        if (out && pool) {
            memcpy((char*)pool + (requiredSize - fragmentLength - 1), fragmentStart,
                   fragmentLength);
            out->fragment = (char*)pool + (requiredSize - fragmentLength - 1);
            out->fragment[fragmentLength] = '\0';
        }
    }

    // Calculate the total required buffer size
    if (require) {
        *require = requiredSize; // Update with actual required size
    }

    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpUriSweepPath(char* dst, const char* src, u64 srcSize) {
    LOG_TRACE(Lib_Http, "called");

    if (!dst || !src) {
        LOG_ERROR(Lib_Http, "Invalid parameters");
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }

    if (srcSize == 0) {
        dst[0] = '\0';
        return ORBIS_OK;
    }

    u64 len = 0;
    while (len < srcSize && src[len] != '\0') {
        len++;
    }

    for (u64 i = 0; i < len; i++) {
        dst[i] = src[i];
    }
    dst[len] = '\0';

    char* read = dst;
    char* write = dst;

    while (*read) {
        if (read[0] == '.' && read[1] == '.' && read[2] == '/') {
            read += 3;
            continue;
        }

        if (read[0] == '.' && read[1] == '/') {
            read += 2;
            continue;
        }

        if (read[0] == '/' && read[1] == '.' && read[2] == '/') {
            read += 2;
            continue;
        }

        if (read[0] == '/' && read[1] == '.' && read[2] == '\0') {
            if (write == dst) {
                *write++ = '/';
            }
            break;
        }

        bool is_dotdot_mid = (read[0] == '/' && read[1] == '.' && read[2] == '.' && read[3] == '/');
        bool is_dotdot_end =
            (read[0] == '/' && read[1] == '.' && read[2] == '.' && read[3] == '\0');

        if (is_dotdot_mid || is_dotdot_end) {
            if (write > dst) {
                if (*(write - 1) == '/') {
                    write--;
                }
                while (write > dst && *(write - 1) != '/') {
                    write--;
                }

                if (is_dotdot_mid && write > dst) {
                    write--;
                }
            }

            if (is_dotdot_mid) {
                read += 3;
            } else {
                break;
            }
            continue;
        }

        if ((read[0] == '.' && read[1] == '\0') ||
            (read[0] == '.' && read[1] == '.' && read[2] == '\0')) {
            break;
        }

        if (read[0] == '/') {
            *write++ = *read++;
        }
        while (*read && *read != '/') {
            *write++ = *read++;
        }
    }

    *write = '\0';
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpUriUnescape(char* out, u64* require, u64 prepare, const char* in) {
    LOG_TRACE(Lib_Http, "called");

    if (!in) {
        LOG_ERROR(Lib_Http, "Invalid input string");
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }

    // Locale-independent hex digit check
    auto IsHex = [](char c) -> bool {
        return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f');
    };

    // Convert hex char to int value
    auto HexToInt = [](char c) -> int {
        if (c >= '0' && c <= '9')
            return c - '0';
        if (c >= 'A' && c <= 'F')
            return c - 'A' + 10;
        if (c >= 'a' && c <= 'f')
            return c - 'a' + 10;
        return 0;
    };

    // Check for valid percent-encoded sequence (%XX)
    auto IsValidPercentSequence = [&](const char* s) -> bool {
        return s[0] == '%' && s[1] != '\0' && s[2] != '\0' && IsHex(s[1]) && IsHex(s[2]);
    };

    u64 needed = 0;
    const char* src = in;
    while (*src) {
        if (IsValidPercentSequence(src)) {
            src += 3;
        } else {
            src++;
        }
        needed++;
    }
    needed++; // null terminator

    if (require) {
        *require = needed;
    }

    if (!out) {
        return ORBIS_OK;
    }

    if (prepare < needed) {
        LOG_ERROR(Lib_Http, "Buffer too small: need {} but only {} available", needed, prepare);
        return ORBIS_HTTP_ERROR_OUT_OF_MEMORY;
    }

    src = in;
    char* dst = out;
    while (*src) {
        if (IsValidPercentSequence(src)) {
            *dst++ = static_cast<char>((HexToInt(src[1]) << 4) | HexToInt(src[2]));
            src += 3;
        } else {
            *dst++ = *src++;
        }
    }
    *dst = '\0';

    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpWaitRequest() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("hvG6GfBMXg8", "libSceHttp", 1, "libSceHttp", sceHttpAbortRequest);
    LIB_FUNCTION("JKl06ZIAl6A", "libSceHttp", 1, "libSceHttp", sceHttpAbortRequestForce);
    LIB_FUNCTION("sWQiqKvYTVA", "libSceHttp", 1, "libSceHttp", sceHttpAbortWaitRequest);
    LIB_FUNCTION("mNan6QSnpeY", "libSceHttp", 1, "libSceHttp", sceHttpAddCookie);
    LIB_FUNCTION("JM58a21mtrQ", "libSceHttp", 1, "libSceHttp", sceHttpAddQuery);
    LIB_FUNCTION("EY28T2bkN7k", "libSceHttp", 1, "libSceHttp", sceHttpAddRequestHeader);
    LIB_FUNCTION("lGAjftanhFs", "libSceHttp", 1, "libSceHttp", sceHttpAddRequestHeaderRaw);
    LIB_FUNCTION("Y1DCjN-s2BA", "libSceHttp", 1, "libSceHttp", sceHttpAuthCacheExport);
    LIB_FUNCTION("zzB0StvRab4", "libSceHttp", 1, "libSceHttp", sceHttpAuthCacheFlush);
    LIB_FUNCTION("wF0KcxK20BE", "libSceHttp", 1, "libSceHttp", sceHttpAuthCacheImport);
    LIB_FUNCTION("A7n9nNg7NBg", "libSceHttp", 1, "libSceHttp",
                 sceHttpCacheRedirectedConnectionEnabled);
    LIB_FUNCTION("nOkViL17ZOo", "libSceHttp", 1, "libSceHttp", sceHttpCookieExport);
    LIB_FUNCTION("seCvUt91WHY", "libSceHttp", 1, "libSceHttp", sceHttpCookieFlush);
    LIB_FUNCTION("pFnXDxo3aog", "libSceHttp", 1, "libSceHttp", sceHttpCookieImport);
    LIB_FUNCTION("Kiwv9r4IZCc", "libSceHttp", 1, "libSceHttp", sceHttpCreateConnection);
    LIB_FUNCTION("qgxDBjorUxs", "libSceHttp", 1, "libSceHttp", sceHttpCreateConnectionWithURL);
    LIB_FUNCTION("6381dWF+xsQ", "libSceHttp", 1, "libSceHttp", sceHttpCreateEpoll);
    LIB_FUNCTION("tsGVru3hCe8", "libSceHttp", 1, "libSceHttp", sceHttpCreateRequest);
    LIB_FUNCTION("rGNm+FjIXKk", "libSceHttp", 1, "libSceHttp", sceHttpCreateRequest2);
    LIB_FUNCTION("Aeu5wVKkF9w", "libSceHttp", 1, "libSceHttp", sceHttpCreateRequestWithURL);
    LIB_FUNCTION("Cnp77podkCU", "libSceHttp", 1, "libSceHttp", sceHttpCreateRequestWithURL2);
    LIB_FUNCTION("0gYjPTR-6cY", "libSceHttp", 1, "libSceHttp", sceHttpCreateTemplate);
    LIB_FUNCTION("Lffcxao-QMM", "libSceHttp", 1, "libSceHttp", sceHttpDbgEnableProfile);
    LIB_FUNCTION("6gyx-I0Oob4", "libSceHttp", 1, "libSceHttp", sceHttpDbgGetConnectionStat);
    LIB_FUNCTION("fzzBpJjm9Kw", "libSceHttp", 1, "libSceHttp", sceHttpDbgGetRequestStat);
    LIB_FUNCTION("VmqSnjZ5mE4", "libSceHttp", 1, "libSceHttp", sceHttpDbgSetPrintf);
    LIB_FUNCTION("KJtUHtp6y0U", "libSceHttp", 1, "libSceHttp", sceHttpDbgShowConnectionStat);
    LIB_FUNCTION("oEuPssSYskA", "libSceHttp", 1, "libSceHttp", sceHttpDbgShowMemoryPoolStat);
    LIB_FUNCTION("L2gM3qptqHs", "libSceHttp", 1, "libSceHttp", sceHttpDbgShowRequestStat);
    LIB_FUNCTION("pxBsD-X9eH0", "libSceHttp", 1, "libSceHttp", sceHttpDbgShowStat);
    LIB_FUNCTION("P6A3ytpsiYc", "libSceHttp", 1, "libSceHttp", sceHttpDeleteConnection);
    LIB_FUNCTION("qe7oZ+v4PWA", "libSceHttp", 1, "libSceHttp", sceHttpDeleteRequest);
    LIB_FUNCTION("4I8vEpuEhZ8", "libSceHttp", 1, "libSceHttp", sceHttpDeleteTemplate);
    LIB_FUNCTION("wYhXVfS2Et4", "libSceHttp", 1, "libSceHttp", sceHttpDestroyEpoll);
    LIB_FUNCTION("1rpZqxdMRwQ", "libSceHttp", 1, "libSceHttp", sceHttpGetAcceptEncodingGZIPEnabled);
    LIB_FUNCTION("aCYPMSUIaP8", "libSceHttp", 1, "libSceHttp", sceHttpGetAllResponseHeaders);
    LIB_FUNCTION("9m8EcOGzcIQ", "libSceHttp", 1, "libSceHttp", sceHttpGetAuthEnabled);
    LIB_FUNCTION("mmLexUbtnfY", "libSceHttp", 1, "libSceHttp", sceHttpGetAutoRedirect);
    LIB_FUNCTION("L-DwVoHXLtU", "libSceHttp", 1, "libSceHttp", sceHttpGetConnectionStat);
    LIB_FUNCTION("+G+UsJpeXPc", "libSceHttp", 1, "libSceHttp", sceHttpGetCookie);
    LIB_FUNCTION("iSZjWw1TGiA", "libSceHttp", 1, "libSceHttp", sceHttpGetCookieEnabled);
    LIB_FUNCTION("xkymWiGdMiI", "libSceHttp", 1, "libSceHttp", sceHttpGetCookieStats);
    LIB_FUNCTION("7j9VcwnrZo4", "libSceHttp", 1, "libSceHttp", sceHttpGetEpoll);
    LIB_FUNCTION("IQOP6McWJcY", "libSceHttp", 1, "libSceHttp", sceHttpGetEpollId);
    LIB_FUNCTION("0onIrKx9NIE", "libSceHttp", 1, "libSceHttp", sceHttpGetLastErrno);
    LIB_FUNCTION("16sMmVuOvgU", "libSceHttp", 1, "libSceHttp", sceHttpGetMemoryPoolStats);
    LIB_FUNCTION("Wq4RNB3snSQ", "libSceHttp", 1, "libSceHttp", sceHttpGetNonblock);
    LIB_FUNCTION("hkcfqAl+82w", "libSceHttp", 1, "libSceHttp", sceHttpGetRegisteredCtxIds);
    LIB_FUNCTION("yuO2H2Uvnos", "libSceHttp", 1, "libSceHttp", sceHttpGetResponseContentLength);
    LIB_FUNCTION("0a2TBNfE3BU", "libSceHttp", 1, "libSceHttp", sceHttpGetStatusCode);
    LIB_FUNCTION("A9cVMUtEp4Y", "libSceHttp", 1, "libSceHttp", sceHttpInit);
    LIB_FUNCTION("hPTXo3bICzI", "libSceHttp", 1, "libSceHttp", sceHttpParseResponseHeader);
    LIB_FUNCTION("Qq8SfuJJJqE", "libSceHttp", 1, "libSceHttp", sceHttpParseStatusLine);
    LIB_FUNCTION("P5pdoykPYTk", "libSceHttp", 1, "libSceHttp", sceHttpReadData);
    LIB_FUNCTION("u05NnI+P+KY", "libSceHttp", 1, "libSceHttp", sceHttpRedirectCacheFlush);
    LIB_FUNCTION("zNGh-zoQTD0", "libSceHttp", 1, "libSceHttp", sceHttpRemoveRequestHeader);
    LIB_FUNCTION("4fgkfVeVsGU", "libSceHttp", 1, "libSceHttp", sceHttpRequestGetAllHeaders);
    LIB_FUNCTION("mSQCxzWTwVI", "libSceHttp", 1, "libSceHttp", sceHttpsDisableOption);
    LIB_FUNCTION("zJYi5br6ZiQ", "libSceHttp", 1, "libSceHttp", sceHttpsDisableOptionPrivate);
    LIB_FUNCTION("f42K37mm5RM", "libSceHttp", 1, "libSceHttp", sceHttpsEnableOption);
    LIB_FUNCTION("I4+4hKttt1w", "libSceHttp", 1, "libSceHttp", sceHttpsEnableOptionPrivate);
    LIB_FUNCTION("1e2BNwI-XzE", "libSceHttp", 1, "libSceHttp", sceHttpSendRequest);
    LIB_FUNCTION("HRX1iyDoKR8", "libSceHttp", 1, "libSceHttp", sceHttpSetAcceptEncodingGZIPEnabled);
    LIB_FUNCTION("qFg2SuyTJJY", "libSceHttp", 1, "libSceHttp", sceHttpSetAuthEnabled);
    LIB_FUNCTION("jf4TB2nUO40", "libSceHttp", 1, "libSceHttp", sceHttpSetAuthInfoCallback);
    LIB_FUNCTION("T-mGo9f3Pu4", "libSceHttp", 1, "libSceHttp", sceHttpSetAutoRedirect);
    LIB_FUNCTION("PDxS48xGQLs", "libSceHttp", 1, "libSceHttp", sceHttpSetChunkedTransferEnabled);
    LIB_FUNCTION("0S9tTH0uqTU", "libSceHttp", 1, "libSceHttp", sceHttpSetConnectTimeOut);
    LIB_FUNCTION("XNUoD2B9a6A", "libSceHttp", 1, "libSceHttp", sceHttpSetCookieEnabled);
    LIB_FUNCTION("pM--+kIeW-8", "libSceHttp", 1, "libSceHttp", sceHttpSetCookieMaxNum);
    LIB_FUNCTION("Kp6juCJUJGQ", "libSceHttp", 1, "libSceHttp", sceHttpSetCookieMaxNumPerDomain);
    LIB_FUNCTION("7Y4364GBras", "libSceHttp", 1, "libSceHttp", sceHttpSetCookieMaxSize);
    LIB_FUNCTION("Kh6bS2HQKbo", "libSceHttp", 1, "libSceHttp", sceHttpSetCookieRecvCallback);
    LIB_FUNCTION("GnVDzYfy-KI", "libSceHttp", 1, "libSceHttp", sceHttpSetCookieSendCallback);
    LIB_FUNCTION("pHc3bxUzivU", "libSceHttp", 1, "libSceHttp", sceHttpSetCookieTotalMaxSize);
    LIB_FUNCTION("8kzIXsRy1bY", "libSceHttp", 1, "libSceHttp",
                 sceHttpSetDefaultAcceptEncodingGZIPEnabled);
    LIB_FUNCTION("22buO-UufJY", "libSceHttp", 1, "libSceHttp", sceHttpSetDelayBuildRequestEnabled);
    LIB_FUNCTION("-xm7kZQNpHI", "libSceHttp", 1, "libSceHttp", sceHttpSetEpoll);
    LIB_FUNCTION("LG1YW1Uhkgo", "libSceHttp", 1, "libSceHttp", sceHttpSetEpollId);
    LIB_FUNCTION("pk0AuomQM1o", "libSceHttp", 1, "libSceHttp", sceHttpSetHttp09Enabled);
    LIB_FUNCTION("i9mhafzkEi8", "libSceHttp", 1, "libSceHttp", sceHttpSetInflateGZIPEnabled);
    LIB_FUNCTION("s2-NPIvz+iA", "libSceHttp", 1, "libSceHttp", sceHttpSetNonblock);
    LIB_FUNCTION("gZ9TpeFQ7Gk", "libSceHttp", 1, "libSceHttp", sceHttpSetPolicyOption);
    LIB_FUNCTION("2NeZnMEP3-0", "libSceHttp", 1, "libSceHttp", sceHttpSetPriorityOption);
    LIB_FUNCTION("i+quCZCL+D8", "libSceHttp", 1, "libSceHttp", sceHttpSetProxy);
    LIB_FUNCTION("mMcB2XIDoV4", "libSceHttp", 1, "libSceHttp", sceHttpSetRecvBlockSize);
    LIB_FUNCTION("yigr4V0-HTM", "libSceHttp", 1, "libSceHttp", sceHttpSetRecvTimeOut);
    LIB_FUNCTION("h9wmFZX4i-4", "libSceHttp", 1, "libSceHttp", sceHttpSetRedirectCallback);
    LIB_FUNCTION("PTiFIUxCpJc", "libSceHttp", 1, "libSceHttp", sceHttpSetRequestContentLength);
    LIB_FUNCTION("vO4B-42ef-k", "libSceHttp", 1, "libSceHttp", sceHttpSetRequestStatusCallback);
    LIB_FUNCTION("K1d1LqZRQHQ", "libSceHttp", 1, "libSceHttp", sceHttpSetResolveRetry);
    LIB_FUNCTION("Tc-hAYDKtQc", "libSceHttp", 1, "libSceHttp", sceHttpSetResolveTimeOut);
    LIB_FUNCTION("a4VsZ4oqn68", "libSceHttp", 1, "libSceHttp", sceHttpSetResponseHeaderMaxSize);
    LIB_FUNCTION("xegFfZKBVlw", "libSceHttp", 1, "libSceHttp", sceHttpSetSendTimeOut);
    LIB_FUNCTION("POJ0azHZX3w", "libSceHttp", 1, "libSceHttp", sceHttpSetSocketCreationCallback);
    LIB_FUNCTION("7WcNoAI9Zcw", "libSceHttp", 1, "libSceHttp", sceHttpsFreeCaList);
    LIB_FUNCTION("gcUjwU3fa0M", "libSceHttp", 1, "libSceHttp", sceHttpsGetCaList);
    LIB_FUNCTION("JBN6N-EY+3M", "libSceHttp", 1, "libSceHttp", sceHttpsGetSslError);
    LIB_FUNCTION("DK+GoXCNT04", "libSceHttp", 1, "libSceHttp", sceHttpsLoadCert);
    LIB_FUNCTION("jUjp+yqMNdQ", "libSceHttp", 1, "libSceHttp", sceHttpsSetMinSslVersion);
    LIB_FUNCTION("htyBOoWeS58", "libSceHttp", 1, "libSceHttp", sceHttpsSetSslCallback);
    LIB_FUNCTION("U5ExQGyyx9s", "libSceHttp", 1, "libSceHttp", sceHttpsSetSslVersion);
    LIB_FUNCTION("zXqcE0fizz0", "libSceHttp", 1, "libSceHttp", sceHttpsUnloadCert);
    LIB_FUNCTION("Ik-KpLTlf7Q", "libSceHttp", 1, "libSceHttp", sceHttpTerm);
    LIB_FUNCTION("V-noPEjSB8c", "libSceHttp", 1, "libSceHttp", sceHttpTryGetNonblock);
    LIB_FUNCTION("fmOs6MzCRqk", "libSceHttp", 1, "libSceHttp", sceHttpTrySetNonblock);
    LIB_FUNCTION("59tL1AQBb8U", "libSceHttp", 1, "libSceHttp", sceHttpUnsetEpoll);
    LIB_FUNCTION("5LZA+KPISVA", "libSceHttp", 1, "libSceHttp", sceHttpUriBuild);
    LIB_FUNCTION("CR-l-yI-o7o", "libSceHttp", 1, "libSceHttp", sceHttpUriCopy);
    LIB_FUNCTION("YuOW3dDAKYc", "libSceHttp", 1, "libSceHttp", sceHttpUriEscape);
    LIB_FUNCTION("3lgQ5Qk42ok", "libSceHttp", 1, "libSceHttp", sceHttpUriMerge);
    LIB_FUNCTION("IWalAn-guFs", "libSceHttp", 1, "libSceHttp", sceHttpUriParse);
    LIB_FUNCTION("mUU363n4yc0", "libSceHttp", 1, "libSceHttp", sceHttpUriSweepPath);
    LIB_FUNCTION("thTS+57zoLM", "libSceHttp", 1, "libSceHttp", sceHttpUriUnescape);
    LIB_FUNCTION("qISjDHrxONc", "libSceHttp", 1, "libSceHttp", sceHttpWaitRequest);
};

} // namespace Libraries::Http
