#!/usr/bin/python
#****************************************************************#
# ScriptName: run_xflow.py
# Author: $SHTERM_REAL_USER@alibaba-inc.com
# Create Date: 2022-12-02 16:17
# Modify Author: $SHTERM_REAL_USER@alibaba-inc.com
# Modify Date: 2022-12-02 16:17
# Function: 
#***************************************************************#

from tornado.httpserver import HTTPServer
import tornado.web
import tornado

from make_app import make_app, on_process_start

router = tornado.routing.RuleRouter([
    tornado.routing.Rule(tornado.routing.PathMatches(r"/.*"), make_app()),
])

num_processes = 20
server = HTTPServer(router, max_header_size=int(1e10), max_body_size=int(1e10))
server.listen(8888)
#util_logger.info("listening on port: {port}, num_processes: {num_processes}".format(
#    port=args.port,
#    num_processes=num_processes
#))
server.start(num_processes=num_processes)

main_loop = tornado.ioloop.IOLoop.current()
main_loop.add_callback(on_process_start)
print('finished starting')
main_loop.start()

