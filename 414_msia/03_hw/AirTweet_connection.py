# AirTweet_connection.py

import AirTweet_pb2
import Authentication_pb2
import socket
import ssl
from datetime import datetime


def send_msg(sock, msg_bytes):
    # header = struct.pack('>H', len(msg_bytes))
    header = int.to_bytes(len(msg_bytes), 2, byteorder='big')
    sock.sendall(header + msg_bytes)
    return


def recv_all(sock, sz):
    bytes = sock.recv(sz)
    while len(bytes) < sz:
        bytes += sock.recv(sz - len(bytes))
    return bytes


def recv_msg(sock):
    header = recv_all(sock, 2)
    msg_len = int.from_bytes(header, byteorder='big')
    return recv_all(sock, msg_len)


def authorize(sock, login, password, stream_name):
    # generate login message
    login_msg = Authentication_pb2.LoginRequest(login=login,
                                                enc_password=password,
                                                stream_name=stream_name)
    # send login message
    print('Sending login message')
    send_msg(sock, login_msg.SerializeToString())
    # receive login-reply message and handle it
    raw_msg = recv_msg(sock)
    login_reply = Authentication_pb2.LoginReply()
    login_reply.ParseFromString(raw_msg)
    if login_reply.connection_status != Authentication_pb2.LoginReply.LoginErrorsEnum.OK:
        raise RuntimeError('Login failed: ' +
                           Authentication_pb2.LoginReply.LoginErrorsEnum.Name(login_reply.connection_status))
    # now we're logged in
    print('Logged in successfully as ', login)


def connect(host, port, login, password, stream_name,
            tweet_event_handler, catch_handler_errors=True):
    '''
    # main function, connects to server and invokes user specified handlers in the event loop
    ## tweet_event_handler is expected to have 2 arguments: tweet_id (str) and text (str)
    ## and to return a list of 3 probabilities:
    ## probabilities of negative, neutral and positive sentiments (non-negative + sum up to 1)
    '''
    result = {'problems': [],
              'n_signals': 0,
              'penalty': None,
              'missed_id': []
              }
    # connect to server & authorize
    print(f'Connecting to {host}:{port}')
    context = ssl.SSLContext(ssl.PROTOCOL_TLS)
    with socket.create_connection((host, port)) as sock:
        with context.wrap_socket(sock, server_hostname=host) as ssock:
        #with sock as ssock:
            # make authorization
            authorize(ssock, login, password, stream_name)
            # event-loop for server messages
            while True:
                raw_msg = recv_msg(ssock)
                event_msg = AirTweet_pb2.Event()
                event_msg.ParseFromString(raw_msg)
                # check errors
                if event_msg.error:
                    problem = f'SERVER SENT: "{event_msg.error}"'
                    print(problem)
                    result['problems'] += (datetime.now(), problem)
                # process message from server
                if event_msg.HasField('tweet_id') and event_msg.HasField('text'):
                    try:
                        tweet_id = event_msg.tweet_id
                        tweet_text = event_msg.text
                        signal = tweet_event_handler(tweet_id, tweet_text)
                        assert len(signal) == 3 and all(p >= 0 for p in signal) and 0.99 < sum(signal) < 1.01
                    except Exception as e:
                        if catch_handler_errors:
                            signal = [0.25, 0.5, 0.25]
                            problem = f'Error inside tweet_event_handler: {e}. Forcing probabilities to {signal}'
                            print('!!!***   WARNING   ***!!!\n', problem, '\n!!!*******************!!!')
                            result['problems'] += (datetime.now(), problem)
                        else:
                            raise
                    if signal:
                        signal_msg = AirTweet_pb2.Signal(tweet_id=tweet_id, probability=signal)
                        send_msg(ssock, signal_msg.SerializeToString())
                        result['n_signals'] += 1
                if event_msg.HasField('stream_end') and event_msg.stream_end:
                    # break from repeat-loop in case server stream ends
                    print('Stream has ended, goodbye!')
                    result['penalty'] = event_msg.penalty
                    result['missed_id'] = list(event_msg.missed_id)
                    print('Some statistics:')
                    print('penalty=', result['penalty'])
                    if len(result['missed_id']) > 0:
                        message("!!! Look for missed tweet ids in result['missed_id'] !!!")
                    if event_msg.HasField('score'):
                        result['score'] = event_msg.score
                        print('Your score is', result['score'], '/ 100')
                    print('Connection closed')
                    print(f'You sent total of {result["n_signals"]} signal(s) to server')
                    return result
