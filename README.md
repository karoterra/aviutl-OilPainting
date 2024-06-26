# AviUtl スクリプト - 油絵

Kuwahara filter により油絵風の加工を施す
[AviUtl](http://spring-fragrance.mints.ne.jp/aviutl/) スクリプトです。

高橋直哉氏の[油絵スクリプト](https://www.nicovideo.jp/watch/sm35695116)の処理速度を向上させたものです。

## 導入方法

0. スクリプトの動作には Microsoft Visual C++ 再頒布可能パッケージがインストールされている必要があります。お使いの PC にまだインストールされていない場合はあらかじめインストールしてください。
   - アーキテクチャは x86 用のものが必要です。
   - バージョンは Visual Studio 2022 に対応したものが必要です。
     - 例えば Microsoft Visual C++ 2015-2022 再頒布可能パッケージがインストールされていれば問題ありません。
   - インストーラのダウンロードやインストール方法については以下を参照してください。
     - [サポートされている最新の Visual C++ 再頒布可能パッケージのダウンロード | Microsoft Learn](https://learn.microsoft.com/ja-JP/cpp/windows/latest-supported-vc-redist?view=msvc-170)
     - [Visual C++ 再頒布可能パッケージ - /AviUtl](https://scrapbox.io/aviutl/Visual_C++_%E5%86%8D%E9%A0%92%E5%B8%83%E5%8F%AF%E8%83%BD%E3%83%91%E3%83%83%E3%82%B1%E3%83%BC%E3%82%B8)
1. [Releases](https://github.com/karoterra/aviutl-OilPainting/releases/)
   から最新版の ZIP ファイルをダウンロードしてください。
2. ZIP ファイルを展開し、以下のファイルを AviUtl 拡張編集の `script` フォルダに配置してください。
   - `KaroterraOilPainting.dll`
   - `油絵.anm`

## 使い方

お好きなオブジェクトにアニメーション効果「油絵」を適用してください。
詳細スクリプトファイル内のコメントや[紹介動画](https://www.nicovideo.jp/watch/sm39051118)を参照してください。

## License

このソフトウェアは MIT ライセンスのもとで公開されます。
詳細は [LICENSE](LICENSE) を参照してください。

使用したライブラリ等については [CREDITS](CREDITS.md) を参照してください。

## Change Log

更新履歴は [CHANGELOG](CHANGELOG.md) を参照してください。
