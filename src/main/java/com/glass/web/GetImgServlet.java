package com.glass.web;

import com.glass.util.ConstValues;

import javax.servlet.*;
import javax.servlet.http.*;
import javax.servlet.annotation.*;
import java.io.*;
import java.net.Socket;
import java.net.UnknownHostException;
import java.text.SimpleDateFormat;

/**
 * 传输的tcp数据格式：（数据长度    图片）-----------数据长度5位，不足用空格代替
 * 注意：因为是一张图片一张图片的刷新，可能会有延迟
 * 先读取5位数据长度
 * 然后根据数据长度读取图片数据
 * 当数据读取完毕之后，break循环，发送数据
 * <p>
 * 跳出阻塞的4中常用方法
 * <p>
 * 1.调用socke的shutdownOutput方法关闭输出流，该方法的文档说明为，将此套接字的输出流置于“流的末尾”，
 * 这样另一端的输入流上的read操作就会返回-1。不能调用socket.getInputStream().close()。这样会导致socket被关闭。
 * 2.约定结束标志，当读到该结束标志时退出不再read。
 * 3.设置超时，会在设置的超时时间到达后抛出SocketTimeoutException异常而不再阻塞。
 * 4.在头部约定好数据的长度。当读取到的长度等于这个长度时就不再继续调用read方法。
 */

/**
 *
 *
 */
@WebServlet("/GetImgServlet")
public class GetImgServlet extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException {
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss:SSS");
        System.out.println(sdf.format(System.currentTimeMillis()));
        /**
         * 这个servlet请求每次接收图片之后都会将socket关闭，所以不需要知道数据长度，只需要每次将接收到的图片数据显示在网页上即可
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
            HttpSession session = request.getSession();
            if (session.getAttribute("socket") == null) {
                System.out.println("还未创建socket，现在创建……");
                socket = new Socket(ConstValues.serverIp, ConstValues.serverColorPort);
                session.setAttribute("socket", socket);
            }
            Object socketObj = session.getAttribute("socket");
            socket = (Socket) socketObj;
            if ((!socket.isConnected()) || socket.isClosed()) {  //如果socket已经关闭，那么重新创建一个
                socket = new Socket(ConstValues.serverIp, ConstValues.serverColorPort);
                session.setAttribute("socket", socket);
            }

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
            //len=bfs.read(size);
            //if(len !=-1){
            //int imgSize = Integer.valueOf(new String(size).strip());
            //System.out.println("图片数据大小：" + imgSize);


            response.setContentType("image/png");
            while ((len = bfs.read(imgs)) != -1) {
                out.write(imgs, 0, len);
                count += len;
            }
            out.flush();

            System.out.println("成功接收图片,实际结束字节数：" + count);
           /* }   else{
                System.out.println("错误，未读取到数据长度");
                response.setCharacterEncoding("utf-8");
                out.println("false");
            }*/

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
