
问题分析：

首先由于问题要求的是预测在用户在2016年7月领取优惠券后15之内是否使用此优惠券，因此可以将问题标记成分类问题，并且是一个二分类问题。

1数据预处理

不属于同一量纲

信息冗余
定性特征不能直接使用
将定性特征转化为定量特征
存在缺失值
信息利用率低
1 特征提取
特征选择时考虑使用过拟合训练的方法，使用100%数据集进行训练，使用100%数据集进行测试，观察auc
当auc距离1越远的时候，说明特征不够多，再次探索更多的特征，直到auc接近1
当过拟合训练完成后，输出特征重要性。删除特征重要性低的特征，不断地过拟合训练，保证auc基本不变
。
首先根据用户的线下优惠券使用情况进行分析，将所有的特征分为五类，用户特征，商户特征，优惠券特征，用户商户组合特征，用户优惠券组合特征，（商户优惠券组合特征）。
用户特征：
	u1领取优惠券的次数
	u2使用优惠券消费的次数
	u3线下领取优惠券但没有使用的次数
	u4优惠券的消费率
	u5线下普通消费次数
	u6平均普通消费时间间隔
	u7平均优惠券消费时间间隔
	u8 = u6/15用户15天内的普通消费平均时间间隔
	u9 = u7/15用户15天内的优惠券消费平均时间间隔 
	u10领取优惠券到使用优惠券的平均时间间隔 
	u11=u10/15 表示在15天内使用掉优惠券的值大小，值越小越有可能，值为0表示可能性最大 
	u19 = u2/u3使用优惠券与没有使用优惠券的比值
	u20 = u2/u25优惠券消费占比
	u21领取优惠券到使用优惠券间隔小于15天的次数 
	u22 = u21/u2表示用户15天使用掉优惠券的次数除以使用优惠券的次数，表示在15天使用掉优惠券的可能，值越大越好
	u23 = u21/u3表示用户15天使用掉优惠券的次数除以领取优惠券未消费的次数，表示在15天使用掉优惠券的可能，值越大越好。
	u24 = u21/u1表示用户15天使用掉优惠券的次数除以领取优惠券的总次数，表示在15天使用掉优惠券的可能，值越大越好。
	u25总消费次数
	u27 消费优惠券的最低折率
	u28核销优惠券的最高消费折率
	u32 用户核销的优惠券的种类数
	u33用户核销过的不同优惠券数量占所有不同优惠券的比重
	u34用户平均每种优惠券核销多少张
	u35核销优惠券用户-商家平均距离
	u36用户核销优惠券中的最小用户-商家距离
	u37用户核销优惠券中的最大用户-商家距离
	u41不同优惠券领取次数
	u42不同优惠券使用次数
	u43不同优惠券不使用次数
	u44不同优惠券使用率
	u45 消费优惠券的平均折率
	u47用户领取所有不同优惠券种类数
	u48满减类型优惠券领取次数
	u49打折类型优惠券领取次数
	优惠券类型

商户特征：m
	商户总的消费笔数：m0
	商户优惠券消费笔数：m1
	商户正常消费笔数：m2
	商户优惠券被领取次数：m3
	商户优惠券被领取后的核销率：m4 = m1/m3
	领取后没有被使用的优惠券：m7
	商户当天发行的优惠券数目：m5
	商户在当天有多少人在此店领券m6
	商家优惠券核销的平均消费折率：m8
	商家优惠券核销的最小消费折率：m9
	商家优惠券核销的最大消费折率：m10
	商家优惠券核销不同的用户数量：m11
	商家优惠券领取不同的用户数量：m12
	核销商家优惠券的不同用户数量其占领取不同的用户比重：m13
	商家优惠券平均每个用户核销多少张：m14
	商家被核销过的不同优惠券数量：m15
	商家领取过的不同优惠券数量的比重：m16
	商家被核销过的不同优惠券数量占所有领取过的不同优惠券数量的比重：m17
	商家被核销优惠券的平均时间：m18
	商家被核销优惠券中的用户-商家平均距离：m19
	商家被核销优惠券中的用户-商家最小距离：m20
	商家被核销优惠券中的用户-商家最大距离：m21
优惠券特征：c
	c1此优惠券一共发行多少张
	c2此优惠券一共被使用多少张
	c3 =c2/c1优惠券使用率
	c4 =c1-c2没有使用的数目
	c5此优惠券在当天发行了多少张
	c6优惠券类型(直接优惠为0, 满减为1)
	c8不同打折优惠券领取次数
	c9不同打折优惠券使用次数
	c10不同打折优惠券不使用次数
	c11=c9/c8不同打折优惠券使用率
	c12优惠券核销平均时间

组合特征：um：用户和商户
	um1用户领取商家的优惠券次数
	um2用户领取商家的优惠券后不核销次数
	um3用户领取商家的优惠券后核销次数
	um4 = um3/ um1用户领取商家的优惠券后核销率
	um5用户对每个商家的不核销次数占用户总的不核销次数的比重
	um6用户在商店总共消费过几次
	um7用户在商店普通消费次数
	um8用户当天在此商店领取的优惠券数目
	um9用户领取优惠券不同商家数量
	um10用户核销优惠券不同商家数量
	um11= um10/ um9用户核销过优惠券的不同商家数量占所有不同商家的比重
	um12= u2/ um9用户平均核销每个商家多少张优惠券

用户与优惠券的组合特征：uc
	uc1用户领取的所有优惠券数目
	uc2用户领取的特定优惠券数目
	uc3用户领取优惠券平均时间间隔
	uc4用户领取特定商家的优惠券数目
	uc5用户领取的不同商家数目
	uc6用户当天领取的优惠券数目
	uc7用户当天领取的特定优惠券数目
	uc8用户领取的所有优惠券种类数目
	uc9商家被领取的优惠券数目
	uc10商家被领取的特定优惠券数目
	uc11商家被多少不同用户领取的数目
	uc12商家发行的所有优惠券种类数目

线上特征
	on_u1用户线上操作次数
	on_u2用户线上点击次数
	on_u3= on_u2/ on_u1用户线上点击率
	on_u4用户线上购买次数
	on_u5= on_u4/ on_u1用户线上购买率
	on_u6用户线上领取次数
	on_u7= on_u6/ on_u1用户线上领取率
	on_u8用户线上不消费次数
	on_u9用户线上优惠券核销次数
	on_u10= on_u9/ on_u6用户线上优惠券核销率
	on_u11=u3/( on_u8+u3)用户线下不消费次数占线上线下总的不消费次数的比重
	on_u12 = u2/( on_u9+u2)用户线下的优惠券核销次数占线上线下总的优惠券核销次数的比重
	on_u13=u1/( on_u6+u1)用户线下领取的记录数量占总的记录数量的比重

共105个特征
