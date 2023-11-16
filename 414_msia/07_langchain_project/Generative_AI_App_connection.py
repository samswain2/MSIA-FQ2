# Generative_AI_App_connection.py

import Generative_AI_App_pb2
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
            question_event_handler, catch_handler_errors=True):
    '''
    # main function, connects to server and invokes user specified handler in the event loop
    ## question_event_handler is expected to have 2 arguments: question_id (int) and question (str)
    ## and to return answer (str)
    '''
    result = {'problems': [],
              'n_signals': 0,
              'penalty': None
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
                event_msg = Generative_AI_App_pb2.ServerMsg()
                event_msg.ParseFromString(raw_msg)
                # check errors
                if event_msg.error:
                    problem = f'SERVER SENT: "{event_msg.error}"'
                    print(problem)
                    result['problems'] += (datetime.now(), problem)
                # process message from server
                if event_msg.HasField('question'):
                    try:
                        question_id = event_msg.question_id
                        question = event_msg.question
                        answer = question_event_handler(question_id, question)
                    except Exception as e:
                        if catch_handler_errors:
                            answer = "Sorry, I don't know"
                            problem = f'Error inside question_event_handler: {e}. Forcing answer to default one: "{answer}"'
                            print('!!!***   WARNING   ***!!!\n', problem, '\n!!!*******************!!!')
                            result['problems'] += (datetime.now(), problem)
                        else:
                            raise
                    if answer:
                        answer_msg = Generative_AI_App_pb2.AnswerMsg(question_id=question_id, answer=answer)
                        send_msg(ssock, answer_msg.SerializeToString())
                        result['n_signals'] += 1
                if event_msg.HasField('stream_end') and event_msg.stream_end:
                    # break from repeat-loop in case server stream ends
                    print('Stream has ended, goodbye!')
                    result['score'] = event_msg.score
                    result['penalty'] = 100 - result['score']
                    print('Some statistics:')
                    print('penalty=', result['penalty'])
                    print('Your score is', result['score'], '/ 100')
                    print('Connection closed')
                    print(f'You sent total of {result["n_signals"]} answer(s) to server')
                    return result
