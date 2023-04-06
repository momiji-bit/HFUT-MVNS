package com.glass.web;

import com.glass.util.ConstValues;

import javax.servlet.*;
import javax.servlet.http.*;
import javax.servlet.annotation.*;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.Socket;
import java.net.UnknownHostException;
import java.text.SimpleDateFormat;

@WebServlet("/getDepthImgServlet")
public class getDepthImgServlet extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException {
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss:SSS");
        System.out.println(sdf.format(System.currentTimeMillis()));
        /**
         * 1.获取socket对象，不存在就创建一个
         * 2.向服务器发送当前要获取的设备编号   (格式：20位设备编号)
         * 3.接收服务器发送的图片
         *
         *      --if(socket不关闭) 可以连续的接收服务器传来的信息，优点：连续性、缺点：图片刷新过快，对网络要求高，一直保持连接对服务器的负载高
         *          严重问题：不关闭的话，需要设置数据格式说明一张图片传输完毕，（图片大小，图片数据），如果某一次数据接收的时候出现错误，会导致往后的数据全是错误的。
         *      --if(socket关闭） 每次请求都会收到服务器 当前正在处理数据的图片  优点：实时性比较好、另外可以根据请求频率确定帧数，稳定性高，即使某一次出现问题也不会影响下一次的连接
         */
        //1.获取socket对象，不存在就创建一个
        Socket socket = null;
        try {
//            HttpSession session = request.getSession();
//            if (session.getAttribute("socketDepth") == null) {
//                System.out.println("还未创建socket，现在创建……");
//                socket = new Socket(ConstValues.serverIp, ConstValues.serverDepthPort);
//                session.setAttribute("socketDepth", socket);
//            }
//            Object socketObj = session.getAttribute("socketDepth");
//            socket = (Socket) socketObj;
//            if ((!socket.isConnected()) || socket.isClosed()) {  //如果socket已经关闭，那么重新创建一个
//                socket = new Socket(ConstValues.serverIp, ConstValues.serverDepthPort);
//                session.setAttribute("socketDepth", socket);
//            }
            socket = new Socket(ConstValues.serverIp, ConstValues.serverDepthPort);
            //2.向服务器发送当前设备的编号
            String deviceId = request.getParameter("deviceId");
            BufferedOutputStream bos = new BufferedOutputStream(socket.getOutputStream());
            byte[] bytes = String.format("%-20s", deviceId).getBytes();
            System.out.println(bytes.length);

            bos.write(bytes);
            bos.flush();


            //3.获取服务器发送来的数据
            InputStream inputStream = socket.getInputStream();
            BufferedInputStream bfs = new BufferedInputStream(inputStream);
            byte[] size = new byte[8];
            byte[] imgs = new byte[1024];
            int count = 0;
            int len = 0;
            ServletOutputStream out = response.getOutputStream();
            len=bfs.read(size);
            System.out.println("size:="+new String(size));
            if(len !=-1){
                int imgSize = Integer.valueOf(new String(size).strip());
                System.out.println("图片数据大小：" + imgSize);


                response.setContentType("image/png");
                while ((len = bfs.read(imgs)) != -1) {
                    out.write(imgs, 0, len);
                    count += len;
                    if(count>=imgSize){
                        System.out.println("图片传输完毕");
                        break;
                    }
                }
                out.flush();
                System.out.println("成功接收图片,实际结束字节数：" + count);
            }   else{
                System.out.println("错误，未读取到数据长度");
                response.setCharacterEncoding("utf-8");
                out.println("false");
            }

        } catch (UnknownHostException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (socket != null) {
                try {
                    socket.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        System.out.println(sdf.format(System.currentTimeMillis()));

    }

    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        this.doGet(request, response);
    }
}
