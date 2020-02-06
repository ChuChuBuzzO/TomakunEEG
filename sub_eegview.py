# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'sub_eegview.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1855, 837)
        Dialog.setStyleSheet("background-color: white")
        self.verticalLayoutWidget = QtWidgets.QWidget(Dialog)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 1841, 811))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalScrollBar = QtWidgets.QScrollBar(Dialog)
        self.verticalScrollBar.setGeometry(QtCore.QRect(1840, 0, 16, 781))
        self.verticalScrollBar.setStyleSheet("background-color: gray")
        self.verticalScrollBar.setOrientation(QtCore.Qt.Vertical)
        self.verticalScrollBar.setObjectName("verticalScrollBar")
        self.PushButtonOption = QtWidgets.QPushButton(Dialog)
        self.PushButtonOption.setGeometry(QtCore.QRect(90, 800, 71, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.PushButtonOption.setFont(font)
        self.PushButtonOption.setStyleSheet("background-color: rgb(229, 204, 255)")
        self.PushButtonOption.setObjectName("PushButtonOption")
        self.PushButtonEvents = QtWidgets.QPushButton(Dialog)
        self.PushButtonEvents.setGeometry(QtCore.QRect(10, 800, 71, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.PushButtonEvents.setFont(font)
        self.PushButtonEvents.setStyleSheet("background-color: rgb(229, 204, 255)")
        self.PushButtonEvents.setObjectName("PushButtonEvents")
        self.horizontalScrollBar = QtWidgets.QScrollBar(Dialog)
        self.horizontalScrollBar.setGeometry(QtCore.QRect(190, 800, 1521, 31))
        self.horizontalScrollBar.setStyleSheet("background-color: gray")
        self.horizontalScrollBar.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalScrollBar.setObjectName("horizontalScrollBar")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.PushButtonOption.setText(_translate("Dialog", "Option"))
        self.PushButtonEvents.setText(_translate("Dialog", "Events"))
