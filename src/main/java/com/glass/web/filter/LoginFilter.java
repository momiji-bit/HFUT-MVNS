package com.glass.web.filter;

import javax.servlet.*;
import javax.servlet.annotation.*;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpSession;
import java.io.IOException;

@WebFilter("*.jsp")
public class LoginFilter implements Filter {
    public void init(FilterConfig config) throws ServletException {
    }

    public void destroy() {
    }

    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain) throws ServletException, IOException {
        HttpServletRequest req =  (HttpServletRequest)request;
        String requestURI = req.getRequestURI();
        System.out.println(requestURI);
        if(requestURI.equals("/pages/login.jsp")||requestURI.equals("/pages/register.jsp")){
            chain.doFilter(req,response);
            return;
        }

        HttpSession session = req.getSession();
        Object username = session.getAttribute("username");
        if(username == null){
            req.setAttribute("msg","您还未登录");
            req.getRequestDispatcher("/pages/login.jsp").forward(req,response);
        } else {
            chain.doFilter(request, response);
        }


    }
}
