package com.glass.web;

import com.glass.mapper.UserMapper;
import com.glass.pojo.Menu;
import com.glass.pojo.TwoMenu;
import com.glass.pojo.User;
import com.glass.util.SqlSessionFactoryUtils;
import org.apache.ibatis.session.SqlSession;
import org.apache.ibatis.session.SqlSessionFactory;

import javax.servlet.*;
import javax.servlet.http.*;
import javax.servlet.annotation.*;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

@WebServlet("/test")
public class test extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        request.setCharacterEncoding("UTF-8");
        String name = request.getParameter("username");
        String pwd = request.getParameter("password");
        System.out.println("name:"+name+"   password:"+pwd);

        SqlSessionFactory sqlSessionFactory = SqlSessionFactoryUtils.getSqlSessionFactory();
        SqlSession sqlSession = sqlSessionFactory.openSession();
        UserMapper mapper = sqlSession.getMapper(UserMapper.class);
        List<User> users = mapper.selectToLogin(name, pwd);


        response.setContentType("text/json;charset=UTF-8");
        response.setCharacterEncoding("UTF-8");
        PrintWriter out = response.getWriter();
        String str;
        if(users.size()>0){
            // 解决json中文乱码
            System.out.println(users.get(0));

            str ="{\"success\":true}";
            HttpSession session = request.getSession();
            session.setAttribute("username",users.get(0).getUsername());
            session.setAttribute("myDeviceId",users.get(0).getDeviceId());

            List<Menu> oneMenuList = new ArrayList<>();
            List<TwoMenu > twoMenuList = new ArrayList<>();
           if("1".equals(users.get(0).getAuthority())){
               //如果是管理员

               //设置二级菜单
               twoMenuList.add(new TwoMenu("设备状态管理","pages/admin/status.jsp"));
               twoMenuList.add(new TwoMenu("设备实时数据","pages/admin/deviceInfo.jsp"));
               //设置一级菜单     开发者调试功能--》设备状态管理、设备实时数据
               oneMenuList.add(new Menu("开发者调试功能",twoMenuList));
                session.setAttribute("init","/statics/layui/api/adminInit.json");

           } else {
               //如果是普通用户
               //设置二级菜单
               twoMenuList.add(new TwoMenu("我的设备","pages/user/myDevice.jsp"));
               //设置一级菜单
               oneMenuList.add(new Menu("设备管理",twoMenuList));
               session.setAttribute("init","/statics/layui/api/userInit.json");
           }
            session.setAttribute("oneMenuList",oneMenuList);
        } else {
            str ="{\"success\":false,\"message\":\"用户名或密码错误\"}";
            System.out.println("登录失败  "+str);
        }
        out.println(str);
        out.flush();
        out.close();
    }

    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        this.doGet(request, response);

    }
}
