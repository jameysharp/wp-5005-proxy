This is a stateless adaptor for WordPress RSS and Atom feeds which
provides their full post history as an [RFC5005][]-compliant feed. It's
useful for reading feeds from sites which have not installed my
[wp-fullhistory][] plugin, which is basically all of them.

[RFC5005]: https://tools.ietf.org/html/rfc5005
[wp-fullhistory]: https://github.com/jameysharp/wp-fullhistory

I wrote it for use with my work-in-progress feed reader, [crawl-rss][],
but I hope it will be useful to others as well.

[crawl-rss]: https://github.com/jameysharp/crawl-rss

This adaptor does not keep any local state, so every time it receives a
request, it sends two GET requests to the origin server. If the request
is for the subscription feed document rather than any of the archive
pages, then it also sends a series of HEAD requests to identify how many
archive pages there are.

For feeds of about 1,000 pages (or 10,000 posts at the WordPress default
setting of 10 posts per page), this takes about 20 requests. (If you can
find a WordPress feed in the wild with more than 1,000 pages of archive
feeds, I want to hear about it!)

As a result, it may be useful to deploy this adaptor with an HTTP cache
in front of it, behind it, or both.

Caching inbound requests avoids re-sending most or all of those requests
and also avoids parsing and hashing the responses. (Configuring a
reverse proxy for this purpose is left as an exercise for the reader.)

Caching outbound requests potentially allows the GET and HEAD requests
to be shared. To do that, set the `HTTP_PROXY` environment variable to
an appropriate URL. For example, if you have [Squid][] running on
localhost, you might set `HTTP_PROXY=http://localhost:3128`.

[Squid]: http://www.squid-cache.org/

If you have to choose one, I think caching inbound requests is a
significantly larger savings.
