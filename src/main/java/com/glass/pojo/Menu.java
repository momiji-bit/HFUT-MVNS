package com.glass.pojo;

import java.util.List;

public class Menu {
    String oneName;
    List<TwoMenu> twoMenuList ;

    public Menu(String oneName, List<TwoMenu> twoMenuList) {
        this.oneName = oneName;
        this.twoMenuList = twoMenuList;
    }

    public String getOneName() {
        return oneName;
    }

    public void setOneName(String oneName) {
        this.oneName = oneName;
    }

    public List<TwoMenu> getTwoMenuList() {
        return twoMenuList;
    }

    public void setTwoMenuList(List<TwoMenu> twoMenuList) {
        this.twoMenuList = twoMenuList;
    }
}
