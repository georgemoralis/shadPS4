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
};

struct HttpConnection {
    int templateId;
    std::string scheme; // "http" or "https"
    std::string host;
    int port;
    bool keepAlive;
    std::unique_ptr<httplib::Client> client;
};

struct HttpRequest {
    int connectionId;
    s32 method;
    std::string path; // path + query, no scheme/host
    u64 contentLength = 0;
    std::vector<std::pair<std::string, std::string>> requestHeaders;

    // Filled by sceHttpSendRequest.
    bool sent = false;
    int statusCode = 0;
    std::string responseBody;
    size_t readOffset = 0;
    std::string responseHeadersBlob;
};

std::mutex g_state_mutex;
std::map<int, HttpTemplate> g_templates;
std::map<int, HttpConnection> g_connections;
std::map<int, HttpRequest> g_requests;
int g_next_template_id = 1;
int g_next_connection_id = 1;
int g_next_request_id = 1;

// HTTP method enum matches OrbisNpWebApiHttpMethod ordering:
// 0=GET, 1=POST, 2=PUT, 3=DELETE, 4=PATCH.
const char* MethodName(s32 method) {
    switch (method) {
    case 0:
        return "GET";
    case 1:
        return "POST";
    case 2:
        return "PUT";
    case 3:
        return "DELETE";
    case 4:
        return "PATCH";
    default:
        return "GET";
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

httplib::Client* GetOrCreateClient(HttpConnection& conn) {
    if (!conn.client) {
        if (conn.scheme == "https") {
            LOG_ERROR(Lib_Http,
                      "HTTPS requested but SSL client not compiled in; "
                      "falling back to plain HTTP for {}:{}",
                      conn.host, conn.port);
        }
        conn.client = std::make_unique<httplib::Client>(conn.host, conn.port);
        conn.client->set_connection_timeout(std::chrono::seconds(kHttpConnectTimeoutSec));
        conn.client->set_read_timeout(std::chrono::seconds(kHttpReadTimeoutSec));
        conn.client->set_keep_alive(conn.keepAlive);
    }
    return conn.client.get();
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

int PS4_SYSV_ABI sceHttpAbortRequest() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpAbortRequestForce() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpAbortWaitRequest() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpAddCookie() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpAddQuery() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
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

int PS4_SYSV_ABI sceHttpAddRequestHeaderRaw() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpAuthCacheExport() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpAuthCacheFlush() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpAuthCacheImport() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpCacheRedirectedConnectionEnabled() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpCookieExport() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpCookieFlush() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpCookieImport() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpCreateConnection() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
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

int PS4_SYSV_ABI sceHttpCreateEpoll() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpCreateRequest() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpCreateRequest2() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
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
    // In Real SDK takes the host from the connection, but the URL passed to
    // CreateRequestWithURL contains a full URL but we use the path part.
    req.path = parsed.path;
    req.contentLength = contentLength;

    LOG_INFO(Lib_Http,
             "sceHttpCreateRequestWithURL: conn={} method={} url='{}' contentLen={} -> req={}",
             connId, MethodName(method), url, contentLength, id);
    return id;
}

int PS4_SYSV_ABI sceHttpCreateRequestWithURL2() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpCreateTemplate() {
    std::scoped_lock lk{g_state_mutex};
    const int id = g_next_template_id++;
    g_templates[id] = HttpTemplate{};
    LOG_DEBUG(Lib_Http, "sceHttpCreateTemplate -> {}", id);
    return id;
}

int PS4_SYSV_ABI sceHttpDbgEnableProfile() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpDbgGetConnectionStat() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpDbgGetRequestStat() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpDbgSetPrintf() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpDbgShowConnectionStat() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpDbgShowMemoryPoolStat() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpDbgShowRequestStat() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpDbgShowStat() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
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

int PS4_SYSV_ABI sceHttpDestroyEpoll() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetAcceptEncodingGZIPEnabled() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
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

int PS4_SYSV_ABI sceHttpGetAuthEnabled() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetAutoRedirect() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetConnectionStat() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetCookie() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetCookieEnabled() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetCookieStats() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetEpoll() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetEpollId() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetLastErrno() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetMemoryPoolStats() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetNonblock() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetRegisteredCtxIds() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetResponseContentLength() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
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
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
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

int PS4_SYSV_ABI sceHttpRedirectCacheFlush() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpRemoveRequestHeader() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpRequestGetAllHeaders() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpsDisableOption() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpsDisableOptionPrivate() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpsEnableOption() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpsEnableOptionPrivate() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
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
        method = MethodName(req.method);
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

    // Pull a Content-Type header out of the request's header list if the
    // caller set one.Fall back to application/json which is what every shadnet endpoint expects.
    std::string contentType = "application/json";
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
    } else if (method == "POST") {
        res = cli->Post(path, httplibHeaders, body, contentType);
    } else if (method == "PUT") {
        res = cli->Put(path, httplibHeaders, body, contentType);
    } else if (method == "PATCH") {
        res = cli->Patch(path, httplibHeaders, body, contentType);
    } else {
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

int PS4_SYSV_ABI sceHttpSetAcceptEncodingGZIPEnabled() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetAuthEnabled() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetAuthInfoCallback() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetAutoRedirect() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetChunkedTransferEnabled() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetConnectTimeOut() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetCookieEnabled() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetCookieMaxNum() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetCookieMaxNumPerDomain() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetCookieMaxSize() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetCookieRecvCallback() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetCookieSendCallback() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetCookieTotalMaxSize() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetDefaultAcceptEncodingGZIPEnabled() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetDelayBuildRequestEnabled() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetEpoll() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetEpollId() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetHttp09Enabled() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetInflateGZIPEnabled() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetNonblock() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetPolicyOption() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetPriorityOption() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetProxy() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetRecvBlockSize() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetRecvTimeOut() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetRedirectCallback() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetRequestContentLength() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetRequestStatusCallback() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetResolveRetry() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetResolveTimeOut() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetResponseHeaderMaxSize() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetSendTimeOut() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetSocketCreationCallback() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpsFreeCaList() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpsGetCaList(int httpCtxId, OrbisHttpsCaList* list) {
    LOG_ERROR(Lib_Http, "(DUMMY) called, httpCtxId = {}", httpCtxId);
    list->certsNum = 0;
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpsGetSslError() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpsLoadCert() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpsSetMinSslVersion() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpsSetSslCallback() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpsSetSslVersion() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpsUnloadCert() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpTerm() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpTryGetNonblock() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpTrySetNonblock() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpUnsetEpoll() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpUriBuild(char* out, u64* require, u64 prepare,
                                 const OrbisHttpUriElement* srcElement, u32 option) {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpUriCopy() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
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
