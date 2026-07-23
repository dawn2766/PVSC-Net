const fs = require("fs");
const path = require("path");
const {
  AlignmentType,
  BorderStyle,
  Document,
  Footer,
  HeadingLevel,
  ImageRun,
  LevelFormat,
  Math,
  MathFraction,
  MathRun,
  MathSubScript,
  MathSubSuperScript,
  MathSum,
  MathSuperScript,
  PageBreak,
  PageNumber,
  Packer,
  Paragraph,
  ShadingType,
  Table,
  TableCell,
  TableRow,
  TabStopPosition,
  TabStopType,
  TextRun,
  VerticalAlign,
  WidthType,
} = require("docx");

const root = path.resolve(__dirname, "..");
const resultsPath = path.join(root, "PVSCNet", "comparisons", "results", "comparison_results.json");
const metricsPath = path.join(root, "PVSCNet", "comparisons", "results");
const preprocessPath = path.join(root, "processed", "preprocess_config.json");
const trainingLogPath = path.join(root, "PVSCNet", "log_best_PVSCNet.json");
const outputDir = path.join(root, "docs", "patent");
const outputPath = path.join(outputDir, "PVSC-Net水下声学目标分类方法发明专利.docx");

const results = JSON.parse(fs.readFileSync(resultsPath, "utf8"));
const preprocess = JSON.parse(fs.readFileSync(preprocessPath, "utf8"));
const trainingLog = JSON.parse(fs.readFileSync(trainingLogPath, "utf8"));

if (preprocess.num_samples !== 4416 || preprocess.num_source_files !== 63) {
  throw new Error("Processed dataset facts do not match the verified experiment.");
}
if (results.class_names.length !== 4) {
  throw new Error("Expected four target classes in comparison_results.json.");
}
if (results.summary.PVSCNet.parameter_count !== 540554) {
  throw new Error("PVSC-Net parameter count changed; regenerate and review the patent.");
}
if (trainingLog.dataset.preprocessing_signature !== "80093138f16f9a9f") {
  throw new Error("PVSC-Net preprocessing signature changed; regenerate the experiments first.");
}

fs.mkdirSync(outputDir, { recursive: true });

const PAGE_WIDTH = 11906;
const PAGE_HEIGHT = 16838;
const CONTENT_WIDTH = 9026;
const BLACK = "000000";
const BORDER_COLOR = "777777";
const HEADER_FILL = "D9EAF7";
const SUBHEADER_FILL = "EAF2F8";
const FONT_BODY = "SimSun";
const FONT_HEADING = "SimHei";

const border = { style: BorderStyle.SINGLE, size: 4, color: BORDER_COLOR };
const borders = { top: border, bottom: border, left: border, right: border };

function run(text, options = {}) {
  return new TextRun({
    text: String(text),
    font: options.font || FONT_BODY,
    size: options.size || 24,
    bold: Boolean(options.bold),
    italics: Boolean(options.italics),
    color: options.color || BLACK,
    break: options.break,
  });
}

function body(text, options = {}) {
  return new Paragraph({
    alignment: options.alignment || AlignmentType.JUSTIFIED,
    indent: options.noIndent ? undefined : { firstLine: 480 },
    spacing: { before: options.before || 0, after: options.after || 120, line: 360 },
    keepNext: Boolean(options.keepNext),
    children: [run(text, options)],
  });
}

function claim(text) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 140, line: 360 },
    children: [run(text)],
  });
}

function mathText(text) {
  return new MathRun(text);
}

function subscript(base, sub) {
  return new MathSubScript({ children: [mathText(base)], subScript: [mathText(sub)] });
}

function superscript(base, superscriptText) {
  return new MathSuperScript({ children: [mathText(base)], superScript: [mathText(superscriptText)] });
}

function subSuperscript(base, sub, superscriptText) {
  return new MathSubSuperScript({
    children: [mathText(base)],
    subScript: [mathText(sub)],
    superScript: [mathText(superscriptText)],
  });
}

function equation(mathChildren, number) {
  return new Paragraph({
    alignment: AlignmentType.LEFT,
    spacing: { before: 120, after: 140, line: 300 },
    tabStops: [
      { type: TabStopType.CENTER, position: CONTENT_WIDTH / 2 },
      { type: TabStopType.RIGHT, position: TabStopPosition.MAX },
    ],
    children: [run("\t"), new Math({ children: mathChildren }), run(`\t（${number}）`)],
  });
}

function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

function heading(text, level = 1, options = {}) {
  const sizes = { 1: 32, 2: 28, 3: 26 };
  return new Paragraph({
    heading: level === 1 ? HeadingLevel.HEADING_1 : level === 2 ? HeadingLevel.HEADING_2 : HeadingLevel.HEADING_3,
    alignment: options.center ? AlignmentType.CENTER : AlignmentType.LEFT,
    spacing: { before: options.before || 220, after: options.after || 160 },
    pageBreakBefore: Boolean(options.pageBreakBefore),
    keepNext: true,
    children: [run(text, { font: FONT_HEADING, size: sizes[level], bold: true })],
  });
}

function sectionLabel(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 100, after: 260 },
    keepNext: true,
    children: [run(text, { font: FONT_HEADING, size: 36, bold: true })],
  });
}

function title(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 300, after: 280 },
    keepNext: true,
    children: [run(text, { font: FONT_HEADING, size: 38, bold: true })],
  });
}

function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 80, after: 200 },
    keepNext: true,
    children: [run(text, { size: 21 })],
  });
}

function imageParagraph(fileName, width, height, altText) {
  const filePath = path.join(metricsPath, fileName);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 80 },
    children: [
      new ImageRun({
        type: "png",
        data: fs.readFileSync(filePath),
        transformation: { width, height },
        altText: { title: altText, description: altText, name: altText },
      }),
    ],
  });
}

function cell(text, width, options = {}) {
  const cellMargin = options.compact ? 45 : 90;
  const lineSpacing = options.compact ? 230 : 280;
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    verticalAlign: VerticalAlign.CENTER,
    shading: options.fill ? { fill: options.fill, type: ShadingType.CLEAR } : undefined,
    margins: { top: cellMargin, bottom: cellMargin, left: 100, right: 100 },
    children: [
      new Paragraph({
        alignment: options.alignment || AlignmentType.CENTER,
        spacing: { before: 0, after: 0, line: lineSpacing },
        children: [run(text, { size: options.size || 20, bold: Boolean(options.bold) })],
      }),
    ],
  });
}

function table(headers, rows, widths, options = {}) {
  if (widths.reduce((sum, value) => sum + value, 0) !== CONTENT_WIDTH) {
    throw new Error("Table widths must sum to the page content width.");
  }
  return new Table({
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({
        tableHeader: true,
        children: headers.map((header, index) =>
          cell(header, widths[index], { fill: HEADER_FILL, bold: true, size: options.fontSize || 20, compact: options.compact })
        ),
      }),
      ...rows.map((row, rowIndex) =>
        new TableRow({
          cantSplit: true,
          children: row.map((value, index) =>
            cell(value, widths[index], {
              fill: rowIndex % 2 === 1 ? "F7F7F7" : undefined,
              size: options.fontSize || 20,
              compact: options.compact,
              alignment: options.leftColumns && options.leftColumns.includes(index)
                ? AlignmentType.LEFT
                : AlignmentType.CENTER,
            })
          ),
        })
      ),
    ],
  });
}

function pct(value, digits = 2) {
  return `${(value * 100).toFixed(digits)}%`;
}

function meanSd(modelName, level, metric) {
  const item = results.summary[modelName][level][metric];
  return `${pct(item.mean)} ± ${pct(item.sample_std)}`;
}

const classTranslation = {
  Cargo: "货船",
  Passengership: "客船",
  Tanker: "油轮",
  Tug: "拖船",
};

const claims = [
  "1、一种基于概率潜变量与多特征融合的水下声学目标分类方法，其特征在于，包括以下步骤：S1，获取具有目标类别标签的多个水下声学源文件，以源文件为不可分割的分组单位构建训练数据集和验证数据集，并将各源文件划分为定长音频窗口；S2，针对每一音频窗口，提取对数梅尔频谱作为主特征，并提取表征频谱纹理、频谱形状、频谱梯度、信号包络及音频统计特性的辅助声学特征；S3，仅基于所述训练数据集中的辅助声学特征拟合特征变换器，利用拟合后的特征变换器将辅助声学特征映射为低维辅助特征；S4，利用多级二维卷积编码器对所述主特征进行编码，对所得特征图分别进行全局平均池化和全局最大池化，并将两种池化结果拼接为主特征向量；利用辅助特征编码器将所述低维辅助特征映射为辅助嵌入向量；S5，将所述主特征向量和所述辅助嵌入向量进行融合，依据融合结果分别生成潜变量均值向量和潜变量对数方差向量；S6，在网络训练阶段，依据所述潜变量均值向量、所述潜变量对数方差向量、标准高斯随机向量和预设噪声比例生成随机潜变量；在网络推理阶段，将所述潜变量均值向量确定为潜变量；S7，将所述潜变量输入分类器，获得水下声学目标的类别输出。",
  "2、根据权利要求1所述的水下声学目标分类方法，其特征在于，步骤S1包括：将各源文件重采样至统一采样率；根据预设窗口长度和窗口重叠率确定规则窗口起点；在不超出源文件有效采样区间的条件下，对除首窗口起点和末窗口起点以外的规则窗口起点施加随机抖动；其中，所述训练数据集与所述验证数据集包含的源文件集合互不相交。",
  "3、根据权利要求1所述的水下声学目标分类方法，其特征在于，所述辅助声学特征包括灰度共生矩阵统计量、Hu不变矩、频谱图梯度统计量、带通滤波信号的解析包络频域统计量，以及均方根、过零率、谱质心和谱带宽。",
  "4、根据权利要求1所述的水下声学目标分类方法，其特征在于，步骤S3中的特征变换器包括依次连接的第一标准化器、主成分分析器、线性判别分析器和第二标准化器；所述第一标准化器、主成分分析器、线性判别分析器及第二标准化器的参数均由所述训练数据集确定，并在验证阶段和推理阶段保持固定。",
  "5、根据权利要求1所述的水下声学目标分类方法，其特征在于，所述多级二维卷积编码器包括四个依次连接的卷积单元；每一卷积单元包括二维卷积层、组归一化层和SiLU激活层，且四个卷积单元的输出通道数逐级增加。",
  "6、根据权利要求1所述的水下声学目标分类方法，其特征在于，所述全局平均池化用于获得所述特征图的整体响应，所述全局最大池化用于获得所述特征图的局部显著响应，所述主特征向量由所述整体响应与所述局部显著响应沿通道维拼接形成。",
  "7、根据权利要求1所述的水下声学目标分类方法，其特征在于，所述辅助特征编码器包括层归一化层、至少两个全连接层及SiLU激活层；所述主特征向量和所述辅助嵌入向量沿特征维拼接后，依次经全连接层、层归一化层、SiLU激活层和Dropout层，获得融合隐藏向量。",
  "8、根据权利要求1所述的水下声学目标分类方法，其特征在于，所述潜变量对数方差向量被限制在预设下限与预设上限之间；在网络训练阶段，先根据受限的潜变量对数方差向量计算潜变量标准差，再将所述潜变量标准差、所述标准高斯随机向量及所述预设噪声比例的乘积与所述潜变量均值向量相加，生成所述随机潜变量。",
  "9、根据权利要求1所述的水下声学目标分类方法，其特征在于，所述分类器包括层归一化层、全连接层、SiLU激活层、Dropout层和类别输出层；所述分类器输出各目标类别对应的未归一化分值，所述网络采用带标签平滑的交叉熵损失进行训练。",
  "10、根据权利要求1所述的水下声学目标分类方法，其特征在于，在每一训练轮次内，按目标类别和源文件实施平衡采样：针对每一目标类别，从该目标类别对应的各源文件中无放回地抽取不超过预设上限的音频窗口，并使不同目标类别参与训练的音频窗口数量相同。",
  "11、根据权利要求1所述的水下声学目标分类方法，其特征在于，针对同一待分类源文件的多个音频窗口分别获得类别输出，并对多个类别输出执行算术平均、置信度加权平均或多数投票，以确定所述待分类源文件的目标类别。",
  "12、一种水下声学目标分类系统，其特征在于，包括源文件分组与窗口生成模块、主特征提取模块、辅助声学特征提取与变换模块、主特征编码模块、辅助特征编码模块、概率潜变量生成模块及分类输出模块；所述各模块被配置为协同执行权利要求1至11中任一项所述的水下声学目标分类方法。",
  "13、一种电子设备，包括处理器和存储器，其特征在于，所述存储器中存储有计算机程序，所述计算机程序由所述处理器执行时，使所述处理器执行权利要求1至11中任一项所述的水下声学目标分类方法。",
  "14、一种计算机可读存储介质，其上存储有计算机程序，其特征在于，所述计算机程序由处理器执行时，实现权利要求1至11中任一项所述的水下声学目标分类方法。",
];

const abstractText =
  "本发明公开一种基于概率潜变量与多特征融合的水下声学目标分类方法、系统、电子设备及存储介质，旨在解决重叠音频窗口随机划分引起的源文件信息泄漏、单一时频特征表征不足以及小样本条件下分类边界稳定性不足的问题。该方法以源文件为分组单位构建互不交叉的训练数据集和验证数据集；并行提取对数梅尔频谱和辅助声学特征，且仅利用训练数据拟合辅助特征变换器；经卷积编码、双全局池化和辅助特征编码后进行特征融合，生成潜变量均值和对数方差。训练阶段按缩放高斯重参数化方式生成随机潜变量，推理阶段以潜变量均值进行确定性分类。该方法适用于船舶目标识别和水下被动声学监测。";

const children = [];

children.push(sectionLabel("说 明 书 摘 要"));
children.push(title("一种基于概率潜变量与多特征融合的水下声学目标分类方法、系统、电子设备及存储介质"));
children.push(body(abstractText));
children.push(body("摘要附图为图1。", { noIndent: true, alignment: AlignmentType.CENTER, before: 160 }));
children.push(imageParagraph("figure_1_pvscnet_architecture.png", 620, 286, "PVSC-Net网络结构摘要附图"));
children.push(caption("图1  PVSC-Net网络结构"));

children.push(pageBreak());
children.push(sectionLabel("权 利 要 求 书"));
claims.forEach((item) => children.push(claim(item)));

children.push(pageBreak());
children.push(sectionLabel("说 明 书"));
children.push(title("一种基于概率潜变量与多特征融合的水下声学目标分类方法、系统、电子设备及存储介质"));

children.push(heading("技术领域", 1));
children.push(body("本发明涉及水下声学信号处理和目标识别技术领域，具体涉及一种将对数梅尔频谱与辅助声学特征进行融合，并利用训练期概率潜变量实现水下声学目标分类的方法、系统、电子设备及计算机可读存储介质。"));

children.push(heading("背景技术", 1));
children.push(body("水下被动声学目标分类通常依据目标辐射噪声中由机械振动、推进器调制和流体动力过程形成的声学模式识别目标类别。受目标航速和载荷变化、传播信道起伏、背景噪声干扰以及可用源文件数量有限等因素影响，同一类别在不同源文件之间可能呈现较大的分布差异，而同一源文件内相邻时间窗口又具有较强相关性。"));
children.push(body("现有处理流程常将连续音频切分为相互重叠的窗口，再按窗口随机划分训练数据和验证数据。该方式可能使同一源文件的相邻窗口分别进入训练数据和验证数据，导致模型在验证阶段利用已见源文件的采集条件或背景特征，所得性能不能准确反映模型对未见源文件的识别能力。因此，数据构建环节需要以源文件为不可分割的分组单位实施隔离。"));
children.push(body("现有基于卷积神经网络的分类方法多以时频图作为单一输入，能够学习局部时频模式，但对频谱灰度共生关系、形状矩、梯度分布、包络调制及基础音频统计量的利用不足。直接拼接高维人工特征又容易引入量纲差异、冗余维度和与类别无关的变化；若标准化或降维参数由训练数据与验证数据共同确定，还会产生特征处理层面的信息泄漏。因此，需要在训练数据范围内完成辅助特征变换，并以低维形式与深度时频特征融合。"));
children.push(body("此外，在源文件数量有限而窗口数量较多的条件下，确定性低维表示容易对训练窗口形成过度适配。训练阶段引入受控随机扰动有助于约束分类器对潜变量局部变化的敏感性，但推理阶段继续随机采样又会导致同一输入对应的输出不稳定。由此，需要一种训练阶段具有随机扰动、推理阶段保持确定输出的潜变量生成机制，并需要限制随机扰动幅度，防止其掩盖有效类别信息。"));
children.push(body("基于上述问题，有必要建立一套贯穿数据分组、主辅特征构建、训练集内特征变换、概率潜变量生成及源文件级输出聚合的水下声学目标分类方法。"));

children.push(heading("发明内容", 1));
children.push(heading("一、要解决的技术问题", 2));
children.push(body("本发明旨在解决以下相互关联的技术问题：其一，重叠窗口按窗口随机划分导致同源信息跨越训练数据和验证数据；其二，单一时频表征不能充分覆盖目标辐射噪声中的纹理、形状、梯度和包络信息，而辅助声学特征直接拼接存在量纲与冗余问题；其三，小样本条件下确定性潜在表示对训练窗口过度适配，而训练和推理阶段均采用随机采样又会造成部署输出波动。"));
children.push(body("为此，本发明提供一种基于概率潜变量与多特征融合的水下声学目标分类方法，在源文件级隔离的前提下，仅以训练数据确定特征处理参数，将深度时频特征与低维辅助声学特征融合，并通过训练阶段受控随机采样、推理阶段均值替代采样的方式兼顾表示稳健性和输出确定性。"));

children.push(heading("二、技术方案", 2));
children.push(body("为实现上述目的，本发明采用如下技术方案。"));
children.push(body("步骤S1，获取水下声学源文件及其目标类别标签，将各源文件重采样至统一采样率。按照窗口长度L和窗口重叠率r确定窗口步长H，在有效采样区间内生成规则窗口起点；固定首窗口起点和末窗口起点，对中间窗口起点加入受约束的随机抖动，并记录每一窗口对应的源文件标识。按目标类别进行分层，以源文件为不可分割的分组单位构建训练数据集和验证数据集，使两数据集的源文件集合互不相交。"));
children.push(equation([
  mathText("H = round[L(1-r)],    "),
  subSuperscript("s", "i", "′"),
  mathText(" = clip("),
  subscript("s", "i"),
  mathText(" + "),
  subscript("δ", "i"),
  mathText(", 1, N-L-1)"),
], "1"));
children.push(body("式中，H为窗口步长，round表示取整，si为第i个规则窗口起点，δi为预设范围内的随机整数，N为源文件采样点数，clip表示区间截断，s'i为抖动后的中间窗口起点。首窗口起点0和末窗口起点N-L保持不变。"));
children.push(body("步骤S2，针对每个音频窗口提取对数梅尔频谱M。先计算短时频谱功率并投影到梅尔滤波器组，再转换到分贝刻度。以M作为主输入，其频带数和时间帧数在同一预处理配置下固定。"));
children.push(equation([
  mathText("M = 10 "),
  subscript("log", "10"),
  mathText("["),
  new MathFraction({
    numerator: [mathText("Mel("), superscript("|STFT(x)|", "2"), mathText(")")],
    denominator: [subscript("P", "ref")],
  }),
  mathText("]"),
], "2"));
children.push(body("步骤S3，与主特征提取并行地提取辅助声学特征a。所述辅助声学特征包括十四维灰度共生矩阵统计量、七维Hu不变矩、四维频谱图梯度统计量、四维包络频域统计量和四维基础音频统计量，共三十三维。包络频域统计量由预设频带内的带通滤波信号经Hilbert变换获得；频谱纹理、形状及梯度统计量由归一化且尺寸固定的频谱图获得。"));
children.push(body("步骤S4，仅使用训练数据集中的辅助声学特征拟合第一标准化器、主成分分析器、线性判别分析器和第二标准化器。将三十三维辅助声学特征经第一标准化器处理后投影至十二维主成分空间，再根据训练标签投影至线性判别空间，并对线性判别输出进行第二次标准化；在四类别实施例中形成三维低维辅助特征。拟合完成后固定全部特征变换参数，用于训练数据、验证数据和待分类数据的统一映射。"));
children.push(equation([
  mathText("a′ = "), subscript("Scale", "2"), mathText("[LDA(PCA("),
  subscript("Scale", "1"), mathText("(a)))]"),
], "3"));
children.push(body("步骤S5，将对数梅尔频谱输入四级二维卷积编码器。各级均采用3×3卷积核、步长2和填充1，输出通道数依次为32、64、128和256；每一卷积层之后连接组归一化层和SiLU激活层。对最后一级特征图分别执行全局平均池化和全局最大池化，获得表征整体响应和局部显著响应的两个256维向量，并沿通道维拼接为512维主特征向量。"));
children.push(equation([
  subscript("h", "main"), mathText(" = GAP("), subscript("F", "4"),
  mathText(") ‖ GMP("), subscript("F", "4"), mathText(")"),
], "4"));
children.push(body("步骤S6，将三维低维辅助特征输入辅助特征编码器，依次执行层归一化、3维至32维线性映射、SiLU激活、32维至32维线性映射、层归一化和SiLU激活，获得32维辅助嵌入向量。将512维主特征向量和32维辅助嵌入向量沿特征维拼接，再依次执行544维至256维线性映射、层归一化、SiLU激活和丢弃概率为0.4的Dropout，获得融合隐藏向量。"));
children.push(body("步骤S7，通过两个相互独立的线性层分别将融合隐藏向量映射为16维潜变量均值向量μ和16维潜变量对数方差向量logvar，并将logvar的各分量限制在区间[-6,2]内。网络训练阶段按照缩放高斯重参数化方式生成潜变量z；网络验证和推理阶段不进行随机采样，直接令z=μ。"));
children.push(equation([
  mathText("σ = exp["),
  new MathFraction({ numerator: [mathText("1")], denominator: [mathText("2")] }),
  mathText(" clip(logvar,-6,2)]"),
], "5"));
children.push(equation([
  mathText("z = μ + λ·ε·σ,    ε ∼ N(0,I)"),
], "6"));
children.push(body("式中，σ为潜变量标准差向量，clip表示逐分量截断，ε为服从标准多元高斯分布N(0,I)的随机向量，λ为非负潜变量噪声比例；具体实施例中λ取0.1，以限制随机扰动幅度。"));
children.push(body("步骤S8，将潜变量依次输入层归一化层、16维至128维线性层、SiLU激活层、丢弃概率为0.4的Dropout层以及128维至目标类别数的输出层，获得各目标类别对应的未归一化分值o。采用标签平滑系数为0.05的交叉熵损失训练网络。"));
children.push(equation([
  subscript("L", "CE"), mathText(" = -"),
  new MathSum({
    subScript: [mathText("k=1")],
    superScript: [mathText("K")],
    children: [subSuperscript("y", "k", "′"), mathText(" log "), subscript("p", "k")],
  }),
  mathText(",    "), subSuperscript("y", "k", "′"), mathText(" = (1-α)"),
  subscript("y", "k"), mathText(" + "),
  new MathFraction({ numerator: [mathText("α")], denominator: [mathText("K")] }),
], "7"));
children.push(body("式中，K为目标类别数，pk为对未归一化分值o执行Softmax运算后得到的第k类预测概率，yk为第k类真实标签分量，y'k为标签平滑后的第k类标签分量，α为标签平滑系数。"));
children.push(body("步骤S9，在部署阶段固定主网络参数和特征变换器参数。针对待分类源文件的每一音频窗口执行步骤S2至S8，获得窗口级类别输出；对属于同一待分类源文件的多个窗口级类别输出执行算术平均、置信度加权平均或多数投票，获得源文件级目标类别。"));

children.push(heading("三、有益效果", 2));
children.push(body("第一，通过以源文件为分组单位划分训练数据和验证数据，并在源文件之间实施窗口限额和平衡采样，避免同一源文件的相关窗口跨数据集分布，降低源文件时长和窗口数量差异对训练梯度及验证结果的影响。"));
children.push(body("第二，通过双全局池化获得时频特征图的整体响应和局部显著响应，并引入频谱纹理、形状、梯度、包络及基础音频统计特性，扩展单一对数梅尔频谱的表征范围；辅助声学特征的标准化和降维参数仅由训练数据确定，从特征处理层面避免验证信息进入训练过程。"));
children.push(body("第三，通过均值向量与对数方差向量参数化潜变量，在训练阶段向潜变量施加幅度受限的随机扰动，使分类器针对潜变量邻域进行优化；在验证和推理阶段直接使用均值向量，保证同一输入对应确定的分类输出。"));
children.push(body("在本申请的三次配对实施例中，PVSC-Net的平均窗口级准确率为80.54%，相较确定性融合对照模型和仅频谱概率对照模型分别提高3.22个百分点和6.37个百分点。上述实验结果用于说明本发明具体实施方式的可实施性及相应技术手段的作用，不构成对本发明保护范围的限定。"));

children.push(heading("附图说明", 1));
children.push(body("图1为本发明PVSC-Net网络结构示意图；", { noIndent: true }));
children.push(body("图2为PVSC-Net与两个受控对照模型的三种子窗口级准确率比较图；", { noIndent: true }));
children.push(body("图3为种子2026实验中PVSC-Net的窗口级混淆矩阵。", { noIndent: true }));

children.push(heading("具体实施方式", 1));
children.push(body("下面结合附图和具体实施例对本发明作进一步说明。所述实施例给出了数据处理、网络构建、训练及验证的一种可行方式，用于说明本发明技术方案能够被实施，不应据此将本发明的保护范围限定为所列参数、目标类别或设备环境。"));

children.push(heading("实施例一：数据构建与无泄漏划分", 2));
children.push(body(`本实施例的原始数据包括${preprocess.num_source_files}个WAV源文件，目标类别为货船、客船、油轮和拖船四类。音频统一重采样为${preprocess.sample_rate} Hz，以${preprocess.clip_duration} s即${preprocess.clip_samples}个采样点为窗口；窗口重叠率为${pct(preprocess.overlap_ratio, 0)}，步长为${preprocess.hop_samples}个采样点即${(preprocess.hop_samples / preprocess.sample_rate).toFixed(2)} s，中间窗口起点抖动范围为正负${preprocess.jitter_samples}个采样点。共生成${preprocess.num_samples}个窗口。`));
children.push(body("针对每一窗口生成64×157的对数梅尔频谱和33维辅助声学特征。数据划分采用按目标类别分层的源文件分组留出方式，验证数据比例设为0.2。不同随机种子对应的验证数据包含12个或13个源文件，且每次划分中的训练源文件集合与验证源文件集合均无交集。对数梅尔频谱的全局最小值和取值范围仅由训练窗口确定；辅助特征变换器仅利用训练数据中按类别和源文件限额平衡抽取的1024个窗口进行拟合。"));
children.push(table(
  ["预处理项目", "实施值", "说明"],
  [
    ["目标类别", "4类", "货船、客船、油轮、拖船"],
    ["源文件/窗口", "63 / 4416", "窗口保留源文件标识"],
    ["采样率", "16000 Hz", "统一重采样"],
    ["窗口", "5 s，重叠75%", "步长1.25 s"],
    ["起点抖动", "±4000采样点", "抖动比例0.2"],
    ["主特征", "64×157", "对数梅尔频谱"],
    ["辅助特征", "33维→PCA12→LDA3", "选择器仅在训练集拟合"],
  ],
  [2100, 2300, 4626],
  { leftColumns: [0, 2] }
));
children.push(caption("表1  数据与预处理参数"));

children.push(heading("实施例二：PVSC-Net网络", 2));
children.push(body("本实施例采用的PVSC-Net结构如图1所示。主特征支路通过四级二维卷积将单通道时频图映射为256通道特征图。各卷积单元采用组归一化，以降低网络对较大批量统计量的依赖。全局平均池化用于汇聚整体响应，全局最大池化用于保留局部显著响应，两者沿通道维拼接形成主特征向量。"));
children.push(body("辅助特征支路将三维低维辅助特征编码为32维辅助嵌入向量，并与512维主特征向量拼接。融合层输出256维融合隐藏向量，潜变量均值分支和潜变量对数方差分支分别输出16维向量。潜变量对数方差分支的权重初始化为零、偏置初始化为-2，以限制训练初始阶段的潜变量标准差。网络总参数量为540554。"));
children.push(table(
  ["模块", "层或运算", "实施参数"],
  [
    ["主编码器1", "Conv2d+GroupNorm+SiLU", "1→32，3×3，步长2，填充1"],
    ["主编码器2", "Conv2d+GroupNorm+SiLU", "32→64，3×3，步长2，填充1"],
    ["主编码器3", "Conv2d+GroupNorm+SiLU", "64→128，3×3，步长2，填充1"],
    ["主编码器4", "Conv2d+GroupNorm+SiLU", "128→256，3×3，步长2，填充1"],
    ["主池化", "全局平均池化||全局最大池化", "256+256=512维"],
    ["辅助编码器", "LayerNorm+Linear+SiLU", "3→32→32"],
    ["融合层", "Linear+LayerNorm+SiLU+Dropout", "544→256，Dropout=0.4"],
    ["概率潜变量", "均值分支/对数方差分支", "256→16；logvar∈[-6,2]"],
    ["分类器", "LayerNorm+Linear+SiLU+Dropout+Linear", "16→128→4，Dropout=0.4"],
  ],
  [1850, 3000, 4176],
  { leftColumns: [0, 1, 2], fontSize: 19 }
));
children.push(caption("表2  PVSC-Net网络参数"));
children.push(imageParagraph("figure_1_pvscnet_architecture.png", 620, 286, "PVSC-Net网络结构"));
children.push(caption("图1  PVSC-Net网络结构"));

children.push(heading("实施例三：训练方法", 2, { pageBreakBefore: true }));
children.push(body("训练采用AdamW优化器和ReduceLROnPlateau学习率调度器。当验证损失连续预设轮次未改善时，将学习率衰减为当前值的0.5倍，最低学习率为初始学习率的0.01倍；梯度范数裁剪阈值设为5.0。每一训练轮次内，按目标类别和源文件实施平衡采样，从每一源文件无放回地抽取不超过128个窗口，并将各目标类别的最终窗口数量统一为各类别可采窗口数量中的最小值。"));
children.push(body("网络训练阶段启用Dropout，并按照式（6）生成随机潜变量；网络验证和部署阶段关闭Dropout，且以潜变量均值向量μ替代随机采样结果。训练35轮，以验证数据窗口级准确率最高的训练轮次作为模型选择结果，并保存相应网络参数；本实施例不启用提前停止。"));
children.push(table(
  ["训练项目", "实施值"],
  [
    ["训练轮数", "35"],
    ["批量大小", "64"],
    ["初始学习率", "3×10^-4"],
    ["优化器/权重衰减", "AdamW / 1×10^-4"],
    ["潜变量维数/噪声比例", "16 / 0.1"],
    ["标签平滑", "0.05"],
    ["梯度裁剪", "5.0"],
    ["源文件窗口上限", "128/轮"],
    ["实验种子", "2024、2025、2026"],
  ],
  [3600, 5426],
  { leftColumns: [0, 1], fontSize: 18, compact: true }
));
children.push(caption("表3  训练超参数"));

children.push(heading("实施例四：对照方法与实验设计", 2));
children.push(body("为分别考察概率潜变量机制和辅助声学特征融合的作用，设置确定性融合模型和仅频谱概率模型两个受控对照。PVSC-Net与两个对照模型采用相同的四级卷积编码器、双全局池化、潜变量维数、分类器、损失函数、优化器、训练轮数、源文件分组划分及平衡采样规则，使模型间差异集中于被考察的技术模块。"));
children.push(body("确定性融合模型保留对数梅尔频谱主特征支路、33维辅助声学特征、特征变换器、辅助特征编码器及融合层，但以单个16维线性映射替代潜变量均值分支、潜变量对数方差分支和随机重参数化过程。该模型参数量为536442，用于对照考察训练期概率潜变量机制。"));
children.push(body("仅频谱概率模型保留四级卷积编码器、双全局池化、潜变量均值与对数方差分支、训练期缩放高斯采样及相同分类器，但不使用33维辅助声学特征、特征变换器、辅助特征编码器和主辅特征拼接。该模型参数量为531108，用于对照考察辅助声学特征融合。"));
children.push(body("采用随机种子2024、2025和2026分别进行三组配对实验。在同一随机种子下，三个模型共享相同的训练源文件与验证源文件划分；不同随机种子同时改变源文件分组划分和网络训练随机状态。每次实验均训练35轮，并选取验证数据窗口级准确率最高的网络参数。三组重复结果用于描述本实施例在不同划分下的表现，不作为统计显著性结论。"));

const seedRows = [];
for (const seed of [2024, 2025, 2026]) {
  for (const modelName of ["PVSCNet", "DeterministicFusionNet", "SpectrogramOnlyPVNet"]) {
    const metric = results.seeds[String(seed)].models[modelName];
    const display = modelName === "PVSCNet"
      ? "PVSC-Net"
      : modelName === "DeterministicFusionNet"
        ? "确定性融合"
        : "仅频谱概率";
    seedRows.push([
      String(seed),
      display,
      String(metric.best_epoch),
      pct(metric.window_level.accuracy),
      pct(metric.window_level.macro_f1),
      `${metric.source_level.source_count} / ${pct(metric.source_level.accuracy)}`,
    ]);
  }
}
children.push(table(
  ["种子", "模型", "最佳轮", "窗口准确率", "窗口Macro-F1", "源文件数/准确率"],
  seedRows,
  [900, 1800, 1000, 1600, 1600, 2126],
  { fontSize: 16, leftColumns: [1], compact: true }
));
children.push(caption("表4  三种子配对实验结果"));

children.push(heading("实施例五：实验结果分析", 2));
const pv = results.summary.PVSCNet;
const det = results.summary.DeterministicFusionNet;
const spec = results.summary.SpectrogramOnlyPVNet;
children.push(body(`三种子平均结果如表5和图2所示。PVSC-Net窗口级准确率为${meanSd("PVSCNet", "window_level", "accuracy")}，窗口级Macro-F1为${meanSd("PVSCNet", "window_level", "macro_f1")}。确定性融合模型相应结果为${meanSd("DeterministicFusionNet", "window_level", "accuracy")}和${meanSd("DeterministicFusionNet", "window_level", "macro_f1")}；仅频谱概率模型相应结果为${meanSd("SpectrogramOnlyPVNet", "window_level", "accuracy")}和${meanSd("SpectrogramOnlyPVNet", "window_level", "macro_f1")}。误差项为三个种子的样本标准差。`));
children.push(body(`按随机种子配对计算，PVSC-Net相对确定性融合模型的窗口级准确率差值依次为${results.summary.DeterministicFusionNet.paired_accuracy_difference_from_pvscnet.values.map((value) => pct(value)).join("、")}，平均提高${pct(results.summary.DeterministicFusionNet.paired_accuracy_difference_from_pvscnet.mean)}；相对仅频谱概率模型的差值依次为${results.summary.SpectrogramOnlyPVNet.paired_accuracy_difference_from_pvscnet.values.map((value) => pct(value)).join("、")}，平均提高${pct(results.summary.SpectrogramOnlyPVNet.paired_accuracy_difference_from_pvscnet.mean)}。在本实施例的受控条件下，上述结果表明概率潜变量机制和辅助声学特征融合均对窗口级分类结果产生积极作用。`));
children.push(body(`源文件级结果通过对同一源文件各窗口的未归一化分值求算术平均后确定。PVSC-Net、确定性融合模型和仅频谱概率模型在三个随机种子下的平均源文件级准确率分别为${pct(pv.source_level.accuracy.mean)}、${pct(det.source_level.accuracy.mean)}和${pct(spec.source_level.accuracy.mean)}。三者结果接近，且每次划分仅包含12个或13个验证源文件，因此本实施例不据此主张源文件级性能具有统计显著差异。`));
children.push(table(
  ["模型", "参数量", "窗口准确率", "窗口Macro-F1", "源文件准确率", "源文件Macro-F1"],
  [
    ["PVSC-Net", String(pv.parameter_count), meanSd("PVSCNet", "window_level", "accuracy"), meanSd("PVSCNet", "window_level", "macro_f1"), meanSd("PVSCNet", "source_level", "accuracy"), meanSd("PVSCNet", "source_level", "macro_f1")],
    ["确定性融合", String(det.parameter_count), meanSd("DeterministicFusionNet", "window_level", "accuracy"), meanSd("DeterministicFusionNet", "window_level", "macro_f1"), meanSd("DeterministicFusionNet", "source_level", "accuracy"), meanSd("DeterministicFusionNet", "source_level", "macro_f1")],
    ["仅频谱概率", String(spec.parameter_count), meanSd("SpectrogramOnlyPVNet", "window_level", "accuracy"), meanSd("SpectrogramOnlyPVNet", "window_level", "macro_f1"), meanSd("SpectrogramOnlyPVNet", "source_level", "accuracy"), meanSd("SpectrogramOnlyPVNet", "source_level", "macro_f1")],
  ],
  [1600, 1200, 1600, 1600, 1600, 1426],
  { fontSize: 16, leftColumns: [0], compact: true }
));
children.push(caption("表5  三种子平均性能（均值±样本标准差）"));
children.push(imageParagraph("figure_2_comparison_results.png", 620, 243, "三种子对比实验结果"));
children.push(caption("图2  PVSC-Net与两个受控对照模型的三种子窗口级准确率"));

const recalls2026 = results.seeds["2026"].models.PVSCNet.window_level.recall_by_class;
children.push(body(`种子2026的PVSC-Net最佳检查点位于第${results.seeds["2026"].models.PVSCNet.best_epoch}轮，窗口级准确率为${pct(results.seeds["2026"].models.PVSCNet.window_level.accuracy)}，Macro-F1为${pct(results.seeds["2026"].models.PVSCNet.window_level.macro_f1)}。四类召回率分别为：货船${pct(recalls2026.Cargo)}、客船${pct(recalls2026.Passengership)}、油轮${pct(recalls2026.Tanker)}、拖船${pct(recalls2026.Tug)}。如图3所示，主要误差来自客船窗口与货船、油轮之间的混淆。`));
children.push(imageParagraph("figure_3_pvscnet_confusion.png", 430, 374, "PVSC-Net种子2026混淆矩阵"));
children.push(caption("图3  PVSC-Net窗口级混淆矩阵（种子2026）"));

children.push(heading("实施例六：部署与变形方式", 2));
children.push(body("部署时，加载经验证确定的网络参数及与该网络参数配套的特征变换器参数。对于单个待分类音频窗口，分别构造64×157的对数梅尔频谱和33维辅助声学特征，利用固定的第一标准化、主成分分析、线性判别分析和第二标准化参数获得三维低维辅助特征；将网络设置为推理状态，以潜变量均值向量作为分类器输入，输出四个目标类别对应的未归一化分值和类别索引。"));
children.push(body("对于连续的待分类源文件，按照与训练阶段一致的窗口长度和窗口步长生成多个音频窗口。可对各窗口的未归一化分值求算术平均并选择最大分量对应的目标类别，也可采用多数投票或置信度加权平均确定源文件级目标类别。源文件级聚合方式不改变窗口级网络的训练过程。"));
children.push(body("在保持源文件级数据隔离、训练数据内特征变换、主辅特征融合以及训练随机而推理确定的潜变量生成机制的条件下，梅尔频带数、卷积通道数、潜变量维数、辅助嵌入维数、窗口长度、目标类别数和潜变量噪声比例可根据采样率、计算资源及目标种类进行调整；组归一化可替换为其他适用于相应批量规模的归一化方式；辅助声学特征还可包含与目标机械振动或调制特性相关的其他统计量。"));
children.push(body("本发明的方法可运行于岸基工作站、船载计算机、水下监听节点、浮标处理终端或云端服务器。系统可进一步包括水听器、模数转换模块、音频缓存模块、告警模块及结果显示模块。"));

children.push(pageBreak());
children.push(sectionLabel("说 明 书 附 图"));
children.push(imageParagraph("figure_1_pvscnet_architecture.png", 640, 295, "图1 PVSC-Net网络结构"));
children.push(caption("图1  PVSC-Net网络结构"));
children.push(imageParagraph("figure_2_comparison_results.png", 640, 251, "图2 三种子对比实验"));
children.push(caption("图2  PVSC-Net与两个受控对照模型的三种子窗口级准确率"));
children.push(imageParagraph("figure_3_pvscnet_confusion.png", 440, 383, "图3 PVSC-Net混淆矩阵"));
children.push(caption("图3  PVSC-Net窗口级混淆矩阵（种子2026）"));

children.push(pageBreak());
children.push(sectionLabel("申请信息填写页（不属于说明书正文）"));
children.push(body("申请人：____________________________", { noIndent: true }));
children.push(body("发明人：____________________________", { noIndent: true }));
children.push(body("联系人及电话：______________________", { noIndent: true }));
children.push(body("代理机构/代理师：___________________", { noIndent: true }));
children.push(body("说明：以上主体信息需由申请人提交前据实填写。正式申报前，建议结合现有技术检索结果，对权利要求保护范围、术语一致性及附图形式进行复核。", { noIndent: true, before: 240 }));

const doc = new Document({
  creator: "",
  title: "PVSC-Net水下声学目标分类方法发明专利",
  subject: "PVSC-Net水下声学目标分类方法",
  description: "发明专利申请文件",
  styles: {
    default: {
      document: {
        run: { font: FONT_BODY, size: 24, color: BLACK },
        paragraph: { spacing: { line: 360, after: 120 } },
      },
    },
    paragraphStyles: [
      {
        id: "Heading1",
        name: "Heading 1",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { font: FONT_HEADING, size: 32, bold: true, color: BLACK },
        paragraph: { spacing: { before: 220, after: 160 }, outlineLevel: 0 },
      },
      {
        id: "Heading2",
        name: "Heading 2",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { font: FONT_HEADING, size: 28, bold: true, color: BLACK },
        paragraph: { spacing: { before: 180, after: 140 }, outlineLevel: 1 },
      },
      {
        id: "Heading3",
        name: "Heading 3",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { font: FONT_HEADING, size: 26, bold: true, color: BLACK },
        paragraph: { spacing: { before: 160, after: 120 }, outlineLevel: 2 },
      },
    ],
  },
  numbering: {
    config: [
      {
        reference: "steps",
        levels: [
          {
            level: 0,
            format: LevelFormat.DECIMAL,
            text: "%1.",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          },
        ],
      },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: PAGE_WIDTH, height: PAGE_HEIGHT },
          margin: { top: 1276, right: 1440, bottom: 1276, left: 1440 },
        },
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [run("第 ", { size: 18 }), new TextRun({ children: [PageNumber.CURRENT], font: FONT_BODY, size: 18 }), run(" 页", { size: 18 })],
            }),
          ],
        }),
      },
      children,
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(outputPath, buffer);
  console.log(outputPath);
  console.log(`bytes=${buffer.length}`);
});